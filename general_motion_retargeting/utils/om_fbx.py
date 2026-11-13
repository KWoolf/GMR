"""Loader for OM_fbx: FBX skeletal motion exported by Organic Motion.

Organic Motion's markerless system writes a binary FBX with a 28-node
skeleton (Hips, Spine, Spine1, Neck, Head, Left/RightShoulder, Arm, ForeArm,
Hand, Left/RightUpLeg, Leg, Ankle_Left/Right, Left/RightFoot and end nodes),
each name suffixed with "_0". The scene is Y-up in centimetres with the
performer facing +Z and their left on +X, the same convention as Axis Neuron,
and the bind pose stored on the bones is a T-pose. The take's first key
repeats that T-pose before a gap, followed by the capture at a fixed rate.

This loader drops the leading T-pose key, expresses each bone's orientation
relative to its T-pose orientation (so every bone is the identity in the
T-pose, as with the Axis Neuron loader) and converts to GMR's Z-up, X-forward
world in metres. IK configs for this format can then express their rotation
offsets as the robot link's orientation when the robot stands in a T-pose.
"""

import re

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from general_motion_retargeting.utils.fbx_reader import load_fbx_skeleton

# Organic Motion (x: left, y: up, z: forward) -> GMR (x: forward, y: left, z: up)
OM_TO_GMR = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)


def _strip_suffix(name):
    return re.sub(r"_\d+$", "", name)


def _drop_leading_reference_key(times):
    """Index of the first capture frame.

    The first key is a T-pose placed ahead of the capture with a gap longer
    than a frame interval. Return 1 when that is the case, otherwise 0.
    """
    if len(times) < 3:
        return 0
    dt = np.median(np.diff(times[1:]))
    return 1 if times[1] - times[0] > 2.0 * dt else 0


def _resample(times, pos, rots, target_fps):
    """Resample global positions (lerp) and orientations (slerp)."""
    t_new = np.arange(times[0], times[-1] + 1e-9, 1.0 / target_fps)
    out_pos = np.empty((len(t_new),) + pos.shape[1:])
    for j in range(pos.shape[1]):
        for k in range(3):
            out_pos[:, j, k] = np.interp(t_new, times, pos[:, j, k])
    out_rots = [Slerp(times, rot)(t_new) for rot in rots]
    return out_pos, out_rots


def load_om_fbx_file(fbx_file, target_fps=None, start=None, end=None):
    """
    Load an Organic Motion FBX export.

    Returns (frames, human_height, fps), where frames is a list of
    {bone_name: [position, orientation_wxyz]} dictionaries in GMR's world
    frame, human_height is the performer's stature in metres measured on the
    T-pose skeleton (top of the skull end node to the lowest foot node), and
    fps is the frame rate of the returned frames. start and end trim the
    capture, in seconds from its first frame, before resampling to
    target_fps.
    """
    skel = load_fbx_skeleton(fbx_file)
    settings = skel.global_settings
    if int(settings.get("UpAxis", 1)) != 1:
        raise ValueError(f"{fbx_file}: expected a Y-up scene, got {settings}")
    to_metres = float(settings.get("UnitScaleFactor", 1.0)) / 100.0

    first = _drop_leading_reference_key(skel.times)
    times = skel.times[first:] - skel.times[first]
    capture_fps = float(np.round(1.0 / np.median(np.diff(times))))

    keep = np.ones(len(times), dtype=bool)
    if start is not None:
        keep &= times >= start - 1e-9
    if end is not None:
        keep &= times <= end + 1e-9
    if keep.sum() < 2:
        raise ValueError(f"Trimming to [{start}, {end}] s leaves fewer than 2 frames")
    idx = np.flatnonzero(keep) + first
    times = skel.times[idx] - skel.times[idx[0]]

    global_pos, global_rot = skel.forward_kinematics(
        skel.translation[idx], skel.rotation[idx]
    )
    rest_pos, rest_rot = skel.forward_kinematics(
        skel.rest_translation[None], skel.rest_rotation[None]
    )

    # Orientation relative to the T-pose, then the change of basis.
    basis = R.from_matrix(OM_TO_GMR)
    rots = [basis * (g * r.inv()) * basis.inv() for g, r in zip(global_rot, rest_rot)]
    pos = global_pos @ OM_TO_GMR.T * to_metres

    fps = capture_fps
    if target_fps is not None and not np.isclose(target_fps, capture_fps):
        pos, rots = _resample(times, pos, rots, target_fps)
        fps = float(target_fps)

    names = [_strip_suffix(n) for n in skel.names]
    quats = np.stack([r.as_quat(scalar_first=True) for r in rots], axis=1)
    frames = []
    for f in range(pos.shape[0]):
        frames.append({name: [pos[f, j], quats[f, j]] for j, name in enumerate(names)})

    up = rest_pos[0, :, 1]
    human_height = (up.max() - up.min()) * to_metres

    return frames, human_height, fps
