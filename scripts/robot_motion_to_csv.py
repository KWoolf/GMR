"""Clean a retargeted robot motion and export it as CSV for mjlab / BeyondMimic.

Two corrections are applied to the GMR pickle before export:

1. Joint velocities are limited to a fraction of the motor limits read from the
   robot's URDF. This removes the single-frame IK branch switches (mostly in the
   shoulders) that a tracking policy cannot follow.
2. The root is raised on frames where a foot mesh goes below the floor.

The CSV has one row per frame: root position (3), root quaternion (xyzw, 4)
and joint positions in the robot model's joint order, which is the format
unitree_rl_mjlab's scripts/csv_to_npz.py reads.
"""

import argparse
import os
import pickle
import re

import mujoco as mj
import numpy as np
from rich import print
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from general_motion_retargeting import ASSET_ROOT, ROBOT_XML_DICT

ROBOT_URDF_DICT = {
    "unitree_r1": ASSET_ROOT / "unitree_r1" / "r1_26dof.urdf",
}

FOOT_BODY_DICT = {
    "unitree_r1": ("left_ankle_roll_link", "right_ankle_roll_link"),
}


def read_velocity_limits(urdf_path):
    text = open(urdf_path).read()
    pattern = r'<joint name="([^"]+)" type="(?:revolute|continuous)">.*?velocity="([\d.eE+-]+)"'
    return {name: float(v) for name, v in re.findall(pattern, text, re.S)}


def limit_joint_velocity(dof_pos, max_step):
    """Clamp per-frame joint changes forwards, then backwards, to max_step."""
    out = dof_pos.copy()
    for i in range(1, len(out)):
        out[i] = out[i - 1] + np.clip(out[i] - out[i - 1], -max_step, max_step)
    for i in range(len(out) - 2, -1, -1):
        out[i] = out[i + 1] + np.clip(out[i] - out[i + 1], -max_step, max_step)
    return out


def foot_mesh_vertices(model, body_names):
    """Collect (geom id, local vertices) for the mesh geoms of the foot bodies."""
    geoms = []
    body_ids = {model.body(name).id for name in body_names}
    for g in range(model.ngeom):
        if (
            model.geom_bodyid[g] in body_ids
            and model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH
        ):
            mesh = model.geom_dataid[g]
            start = model.mesh_vertadr[mesh]
            geoms.append((g, model.mesh_vert[start : start + model.mesh_vertnum[mesh]]))
    return geoms


def lowest_foot_point(model, data, qpos, geoms):
    data.qpos[:] = qpos
    mj.mj_kinematics(model, data)
    lowest = np.inf
    for g, verts in geoms:
        world = data.geom_xpos[g] + verts @ data.geom_xmat[g].reshape(3, 3).T
        lowest = min(lowest, world[:, 2].min())
    return lowest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--motion_file", required=True, help="GMR pickle to clean.")
    parser.add_argument(
        "--robot", default="unitree_r1", choices=sorted(ROBOT_URDF_DICT)
    )
    parser.add_argument("--csv_path", required=True, help="Path of the CSV to write.")
    parser.add_argument(
        "--velocity_fraction",
        type=float,
        default=0.8,
        help="Fraction of each motor's URDF velocity limit to allow.",
    )
    parser.add_argument(
        "--ground_clearance",
        type=float,
        default=0.0,
        help="Height above the floor to keep the lowest foot point at, in metres.",
    )
    parser.add_argument(
        "--segment",
        nargs=2,
        type=float,
        action="append",
        metavar=("START", "END"),
        help="Export only this time range, in seconds of the take. Repeat to export "
        "several segments; each is written to <csv_path stem>_<start>-<end>s.csv.",
    )
    args = parser.parse_args()

    with open(args.motion_file, "rb") as f:
        motion = pickle.load(f)
    fps = float(motion["fps"])
    root_pos = np.array(motion["root_pos"], dtype=float)
    root_rot_xyzw = np.array(motion["root_rot"], dtype=float)
    dof_pos = np.array(motion["dof_pos"], dtype=float)

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    data = mj.MjData(model)
    joint_names = [
        model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] != 0
    ]
    assert len(joint_names) == dof_pos.shape[1], (len(joint_names), dof_pos.shape)

    limits = read_velocity_limits(ROBOT_URDF_DICT[args.robot])
    max_step = np.array([limits[n] for n in joint_names]) * args.velocity_fraction / fps
    speed_before = np.abs(np.diff(dof_pos, axis=0)) * fps
    over_before = int((np.abs(np.diff(dof_pos, axis=0)) > max_step).any(axis=1).sum())
    dof_pos = limit_joint_velocity(dof_pos, max_step)
    speed_after = np.abs(np.diff(dof_pos, axis=0)) * fps
    print(
        f"Velocity limiting: {over_before} of {len(dof_pos) - 1} frame steps were over the limit; "
        f"peak joint speed {speed_before.max():.1f} -> {speed_after.max():.1f} rad/s"
    )

    geoms = foot_mesh_vertices(model, FOOT_BODY_DICT[args.robot])
    qpos = np.zeros((len(dof_pos), model.nq))
    qpos[:, :3] = root_pos
    qpos[:, 3:7] = root_rot_xyzw[:, [3, 0, 1, 2]]
    qpos[:, 7:] = dof_pos
    lowest = np.array([lowest_foot_point(model, data, q, geoms) for q in qpos])
    lift = np.maximum(0.0, args.ground_clearance - lowest)
    # Spread each lift over neighbouring frames so the root does not jitter.
    window = max(1, int(round(0.1 * fps)))
    lift = uniform_filter1d(maximum_filter1d(lift, size=2 * window + 1), size=window)
    root_pos[:, 2] += lift
    print(
        f"Ground correction: lowest foot point {lowest.min() * 100:.1f} cm, "
        f"root raised on {int((lift > 1e-4).sum())} frames by up to {lift.max() * 100:.1f} cm"
    )

    # Segments are cut after cleaning so the velocity limit and ground
    # correction see the motion on both sides of each cut.
    csv = np.concatenate([root_pos, root_rot_xyzw, dof_pos], axis=1)
    duration = (len(csv) - 1) / fps
    outputs = []
    if args.segment is None:
        outputs.append((args.csv_path, csv))
    else:
        stem, ext = os.path.splitext(args.csv_path)
        for start, end in args.segment:
            if not 0.0 <= start < end <= duration + 1e-9:
                raise ValueError(
                    f"Segment {start}-{end} s is outside the take (0-{duration:.2f} s)"
                )
            first = int(round(start * fps))
            last = int(round(end * fps))
            outputs.append(
                (f"{stem}_{start:g}-{end:g}s{ext or '.csv'}", csv[first : last + 1])
            )

    os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)
    for path, rows in outputs:
        np.savetxt(path, rows, delimiter=",", fmt="%.6f")
        print(
            f"Saved {len(rows)} frames ({(len(rows) - 1) / fps:.2f} s) at {fps:g} fps to {path}"
        )
    print("Joint order: " + ", ".join(joint_names))
