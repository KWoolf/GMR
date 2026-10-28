"""Loader for BVH files exported from Noitom Axis Neuron (Perception Neuron).

Axis Neuron exports a Y-up skeleton in centimetres that faces +Z in its rest
T-pose, with every bone's rest orientation equal to the identity. This loader
converts it to GMR's Z-up, X-forward world in metres and conjugates the
orientations by the same change of basis, so each bone's orientation is still
the identity in the T-pose. IK configs for this format can then express their
rotation offsets as the robot link's orientation when the robot stands in the
same T-pose.
"""

import re

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

# Axis Neuron (x: left, y: up, z: forward) -> GMR (x: forward, y: left, z: up)
AXIS_TO_GMR = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)


# Axis Neuron writes Frame Time rounded to three decimals, so 120 Hz appears as
# 0.008 s (125 Hz). Snap to the nearest rate the Perception Neuron hardware
# records at.
NEURON_CAPTURE_RATES = (60.0, 90.0, 96.0, 100.0, 120.0, 240.0)


def _capture_rate(frame_time):
    return min(NEURON_CAPTURE_RATES, key=lambda rate: abs(1.0 / rate - frame_time))


def _read_frame_time(bvh_file):
    with open(bvh_file, "r") as f:
        for line in f:
            match = re.match(r"\s*Frame Time:\s+([\d\.eE\-]+)", line)
            if match:
                return float(match.group(1))
    raise ValueError(f"No 'Frame Time' line in {bvh_file}")


def _read_end_site(bvh_file, joint_name):
    """Return the End Site offset directly under joint_name, in file units."""
    with open(bvh_file, "r") as f:
        lines = f.readlines()
    in_joint = False
    for i, line in enumerate(lines):
        if re.match(rf"\s*JOINT\s+{joint_name}\b", line):
            in_joint = True
        elif in_joint and re.match(r"\s*(JOINT|ROOT)\b", line):
            break
        elif in_joint and "End Site" in line:
            for follow in lines[i + 1 : i + 4]:
                match = re.match(r"\s*OFFSET\s+(\S+)\s+(\S+)\s+(\S+)", follow)
                if match:
                    return np.array(list(map(float, match.groups())))
    return np.zeros(3)


def _rest_height(data, bone):
    """Height of a joint in the rest pose, summed along its parent chain."""
    j = list(data.bones).index(bone)
    height = 0.0
    while j != -1:
        height += data.offsets[j][1]
        j = data.parents[j]
    return height


def _resample(quats, pos, src_fps, tgt_fps):
    """Resample local joint rotations (slerp) and positions (lerp)."""
    n_src = quats.shape[0]
    t_src = np.arange(n_src) / src_fps
    t_tgt = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / tgt_fps)
    out_q = np.empty((len(t_tgt),) + quats.shape[1:])
    for j in range(quats.shape[1]):
        rots = R.from_quat(quats[:, j], scalar_first=True)
        out_q[:, j] = Slerp(t_src, rots)(t_tgt).as_quat(scalar_first=True)
    out_p = np.empty((len(t_tgt),) + pos.shape[1:])
    for j in range(pos.shape[1]):
        for k in range(3):
            out_p[:, j, k] = np.interp(t_tgt, t_src, pos[:, j, k])
    return out_q, out_p


def load_axis_bvh_file(bvh_file, target_fps=None, source_fps=None):
    """
    Load an Axis Neuron BVH export.

    Returns (frames, human_height, fps), where frames is a list of
    {bone_name: [position, orientation_wxyz]} dictionaries in GMR's world
    frame, human_height is the performer's rest-pose stature in metres
    (top of the Head end site above the soles), and fps is the frame rate of
    the returned frames. When target_fps is given, the motion is resampled to
    that rate before forward kinematics. source_fps overrides the capture rate
    inferred from the file's Frame Time.
    """
    data = read_bvh(bvh_file)
    if source_fps is None:
        source_fps = _capture_rate(_read_frame_time(bvh_file))
    src_fps = float(source_fps)

    quats, pos = data.quats, data.pos
    fps = src_fps
    if target_fps is not None and not np.isclose(target_fps, src_fps):
        quats, pos = _resample(quats, pos, src_fps, target_fps)
        fps = float(target_fps)

    global_rot, global_pos = utils.quat_fk(quats, pos, data.parents)

    basis = R.from_matrix(AXIS_TO_GMR)
    basis_quat = basis.as_quat(scalar_first=True)
    basis_inv_quat = basis.inv().as_quat(scalar_first=True)
    global_rot = utils.quat_mul(utils.quat_mul(basis_quat, global_rot), basis_inv_quat)
    global_pos = global_pos @ AXIS_TO_GMR.T / 100.0  # cm to m

    frames = []
    for frame in range(global_pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            result[bone] = [global_pos[frame, i], global_rot[frame, i]]
        frames.append(result)

    # Stature from the rest skeleton: top of the head to the soles.
    head_top = _rest_height(data, "Head") + _read_end_site(bvh_file, "Head")[1]
    sole = _rest_height(data, "LeftFoot") + _read_end_site(bvh_file, "LeftFoot")[1]
    human_height = (head_top - sole) / 100.0  # cm to m

    return frames, human_height, fps
