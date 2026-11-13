"""OM_fbx loading tests on a small synthetic binary FBX file."""

import struct

import numpy as np

from general_motion_retargeting.utils.fbx_reader import (
    FBX_MAGIC,
    FBX_TICKS_PER_SECOND,
    load_fbx_skeleton,
)
from general_motion_retargeting.utils.om_fbx import load_om_fbx_file


def _prop(value):
    if isinstance(value, str):
        data = value.encode()
        return b"S" + struct.pack("<I", len(data)) + data
    if isinstance(value, float):
        return b"D" + struct.pack("<d", value)
    if isinstance(value, np.ndarray):
        fmt = "d" if value.dtype == np.float64 else "q"
        data = value.astype("<" + fmt).tobytes()
        code = b"d" if fmt == "d" else b"l"
        return code + struct.pack("<III", len(value), 0, len(data)) + data
    if isinstance(value, int) and abs(value) < 2**31:
        return b"I" + struct.pack("<i", value)
    return b"L" + struct.pack("<q", value)


def _node(name, props=(), children=(), offset=0):
    """Encode an FBX 7.3 node that starts at byte offset in the file."""
    prop_bytes = b"".join(_prop(p) for p in props)
    head = 13 + len(name)
    body = prop_bytes
    pos = offset + head + len(prop_bytes)
    for child in children:
        encoded = child(pos)
        body += encoded
        pos += len(encoded)
    if children:
        body += b"\x00" * 13
    end = offset + head + len(body)
    header = struct.pack("<III", end, len(props), len(prop_bytes))
    return header + bytes([len(name)]) + name.encode() + body


def node(name, props=(), children=()):
    return lambda offset: _node(name, props, children, offset)


def p70(**values):
    rows = []
    for key, value in values.items():
        name = key.replace("_", " ")
        if isinstance(value, (list, tuple)):
            rows.append(node("P", [name, "Lcl", "", "A", *map(float, value)]))
        else:
            rows.append(node("P", [name, "int", "Integer", "", value]))
    return node("Properties70", [], rows)


def write_fbx(path):
    """Hips_0 (root) and LeftArm_0, in the Organic Motion conventions.

    The rest pose has the bones pointing along +X (the performer's left). Key
    0 repeats the rest pose 0.5 s before the capture; the capture holds the
    rest pose for one frame, then raises the arm 90 degrees about Z.
    """
    hips, arm = 100, 200
    models = [
        node(
            "Model",
            [hips, "Hips_0\x00\x01Model", "Root"],
            [
                p70(Lcl_Translation=(0, 90, 0), Lcl_Rotation=(0, 0, 0)),
            ],
        ),
        node(
            "Model",
            [arm, "LeftArm_0\x00\x01Model", "LimbNode"],
            [
                p70(Lcl_Translation=(20, 50, 0), Lcl_Rotation=(0, 0, 0)),
            ],
        ),
    ]
    times = (np.array([-0.5, 0.0, 0.1, 0.2]) * FBX_TICKS_PER_SECOND).astype(np.int64)
    curves, conns = (
        [],
        [
            node("C", ["OO", hips, 0]),
            node("C", ["OO", arm, hips]),
            node("C", ["OP", 300, arm, "Lcl Rotation"]),
        ],
    )
    curve_nodes = [node("AnimationCurveNode", [300, "R\x00\x01AnimCurveNode", ""])]
    for axis, values in enumerate(([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 90, 90])):
        uid = 400 + axis
        curves.append(
            node(
                "AnimationCurve",
                [uid, "\x00\x01AnimCurve", ""],
                [
                    node("KeyTime", [times]),
                    node("KeyValueDouble", [np.array(values, dtype=np.float64)]),
                ],
            )
        )
        conns.append(node("C", ["OP", uid, 300, "d|" + "XYZ"[axis]]))

    top = [
        node("GlobalSettings", [], [p70(UpAxis=1)]),
        node("Objects", [], models + curve_nodes + curves),
        node("Connections", [], conns),
    ]
    header = FBX_MAGIC + b"\x1a\x00" + struct.pack("<I", 7300)
    data, pos = header, len(header)
    for n in top:
        encoded = n(pos)
        data += encoded
        pos += len(encoded)
    path.write_bytes(data + b"\x00" * 13)


def test_fbx_reader_hierarchy_and_curves(tmp_path):
    path = tmp_path / "two_bones.fbx"
    write_fbx(path)
    skel = load_fbx_skeleton(path)
    assert skel.names == ["Hips_0", "LeftArm_0"]
    assert skel.parents.tolist() == [-1, 0]
    np.testing.assert_allclose(skel.times, [-0.5, 0.0, 0.1, 0.2])
    np.testing.assert_allclose(skel.rotation[:, 1, 2], [0, 0, 90, 90])


def test_om_fbx_drops_tpose_key_and_converts_axes(tmp_path):
    path = tmp_path / "two_bones.fbx"
    write_fbx(path)
    frames, height, fps = load_om_fbx_file(path)
    assert fps == 10.0
    assert len(frames) == 3  # leading T-pose key dropped
    # Y-up centimetres, left on +X -> Z-up metres, left on +Y.
    np.testing.assert_allclose(frames[0]["Hips"][0], [0.0, 0.0, 0.9])
    np.testing.assert_allclose(frames[0]["LeftArm"][0], [0.0, 0.2, 1.4])
    # Identity in the rest pose, then 90 degrees about the old Z (forward),
    # which is GMR's X.
    np.testing.assert_allclose(frames[0]["LeftArm"][1], [1, 0, 0, 0], atol=1e-9)
    s = np.sqrt(0.5)
    np.testing.assert_allclose(frames[2]["LeftArm"][1], [s, s, 0, 0], atol=1e-9)
    np.testing.assert_allclose(height, 0.5)
