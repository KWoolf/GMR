"""Minimal reader for skeletal animation in binary FBX files.

This parses the binary FBX node tree (versions 7.x) with numpy and zlib only,
so no Autodesk FBX SDK is needed. It extracts the skeleton hierarchy, each
bone's static local transform and its baked translation and rotation curves,
and evaluates global joint positions and orientations by forward kinematics.

Only what motion capture exports use is supported: translation, Euler
rotation in any of the six orders, and pre- and post-rotation. Rotation and
scaling pivots, offsets and non-unit scaling are not applied. ASCII FBX files
are rejected.
"""

import struct
import zlib
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

FBX_MAGIC = b"Kaydara FBX Binary  \x00"
# FBX time unit: ticks per second.
FBX_TICKS_PER_SECOND = 46186158000
# FBX RotationOrder enum -> scipy extrinsic Euler sequence. FBX "XYZ" rotates
# about X first, then Y, then Z, all about the parent's axes.
ROTATION_ORDERS = ("xyz", "xzy", "yzx", "yxz", "zxy", "zyx")

_ARRAY_FORMATS = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "B"}
_SCALAR_FORMATS = {"Y": "h", "C": "?", "I": "i", "F": "f", "D": "d", "L": "q"}


@dataclass
class FbxNode:
    name: str
    props: list
    children: list

    def find(self, name):
        for child in self.children:
            if child.name == name:
                return child
        return None

    def find_all(self, name):
        return [child for child in self.children if child.name == name]


def _read_node(f, version):
    if version >= 7500:
        end, n_props, _ = struct.unpack("<QQQ", f.read(24))
    else:
        end, n_props, _ = struct.unpack("<III", f.read(12))
    name = f.read(f.read(1)[0]).decode("latin1")
    if end == 0:
        return None  # null record that closes a node list
    props = []
    for _ in range(n_props):
        code = f.read(1).decode("latin1")
        if code in _SCALAR_FORMATS:
            fmt = "<" + _SCALAR_FORMATS[code]
            props.append(struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0])
        elif code in _ARRAY_FORMATS:
            length, encoding, comp_len = struct.unpack("<III", f.read(12))
            data = f.read(comp_len)
            if encoding == 1:
                data = zlib.decompress(data)
            props.append(np.frombuffer(data, dtype="<" + _ARRAY_FORMATS[code]))
        elif code in ("S", "R"):
            data = f.read(struct.unpack("<I", f.read(4))[0])
            props.append(data.decode("utf-8", "replace") if code == "S" else data)
        else:
            raise ValueError(f"Unknown FBX property type {code!r} in node {name}")
    children = []
    while f.tell() < end:
        child = _read_node(f, version)
        if child is None:
            break
        children.append(child)
    f.seek(end)
    return FbxNode(name, props, children)


def read_fbx_tree(path):
    """Parse a binary FBX file into its top-level nodes, keyed by name."""
    with open(path, "rb") as f:
        if f.read(len(FBX_MAGIC)) != FBX_MAGIC:
            raise ValueError(
                f"{path} is not a binary FBX file (ASCII FBX is not supported)"
            )
        f.read(2)
        version = struct.unpack("<I", f.read(4))[0]
        top = {}
        while True:
            node = _read_node(f, version)
            if node is None:
                break
            top[node.name] = node
    return version, top


def _properties70(node):
    """Read a Properties70 block as {name: value(s)}."""
    out = {}
    block = node.find("Properties70")
    if block is None:
        return out
    for p in block.find_all("P"):
        values = p.props[4:]
        out[p.props[0]] = values[0] if len(values) == 1 else np.array(values, float)
    return out


@dataclass
class FbxSkeleton:
    """Skeleton and animation of the first take of an FBX file.

    names: bone names, parents before children.
    parents: index of each bone's parent, -1 for roots.
    times: frame times in seconds, shape (F,).
    rest_translation, rest_rotation: static local transforms, shape (J, 3),
        rotation in degrees.
    translation, rotation: animated local transforms, shape (F, J, 3).
    pre_rotation, post_rotation: per-bone degrees, shape (J, 3).
    rotation_order: scipy Euler sequence per bone.
    global_settings: the GlobalSettings properties (axes, unit scale).
    """

    names: list
    parents: np.ndarray
    times: np.ndarray
    rest_translation: np.ndarray
    rest_rotation: np.ndarray
    translation: np.ndarray
    rotation: np.ndarray
    pre_rotation: np.ndarray
    post_rotation: np.ndarray
    rotation_order: list
    global_settings: dict

    def forward_kinematics(self, translation=None, rotation=None):
        """Global positions (F, J, 3) and scipy Rotations, one per bone.

        Defaults to the animated pose; pass the rest arrays with a leading
        frame axis to evaluate the static pose.
        """
        if translation is None:
            translation, rotation = self.translation, self.rotation
        n_frames, n_bones, _ = translation.shape
        global_pos = np.zeros((n_frames, n_bones, 3))
        global_rot = [None] * n_bones
        for j in range(n_bones):
            local = (
                R.from_euler("xyz", self.pre_rotation[j], degrees=True)
                * R.from_euler(self.rotation_order[j], rotation[:, j], degrees=True)
                * R.from_euler("xyz", self.post_rotation[j], degrees=True).inv()
            )
            parent = self.parents[j]
            if parent < 0:
                global_rot[j] = local
                global_pos[:, j] = translation[:, j]
            else:
                global_rot[j] = global_rot[parent] * local
                global_pos[:, j] = global_pos[:, parent] + global_rot[parent].apply(
                    translation[:, j]
                )
        return global_pos, global_rot


def _curve(node):
    times = node.find("KeyTime").props[0].astype(np.float64) / FBX_TICKS_PER_SECOND
    values = node.find("KeyValueFloat") or node.find("KeyValueDouble")
    return times, values.props[0].astype(np.float64)


def load_fbx_skeleton(path):
    """Read the skeleton (LimbNode/Root/Null models) and its animation curves."""
    _, top = read_fbx_tree(path)
    objects = {node.props[0]: node for node in top["Objects"].children}
    connections = [c.props for c in top["Connections"].find_all("C")]

    models = {
        uid: node
        for uid, node in objects.items()
        if node.name == "Model" and node.props[2] in ("LimbNode", "Root", "Null")
    }
    parent_of = {}
    for kind, child, parent, *_ in connections:
        if kind == "OO" and child in models and parent in models:
            parent_of[child] = parent

    # Order bones so that parents come before children.
    order = []

    def visit(uid):
        order.append(uid)
        for child in models:
            if parent_of.get(child) == uid:
                visit(child)

    for uid in models:
        if uid not in parent_of:
            visit(uid)

    names = [models[uid].props[1].split("\x00")[0] for uid in order]
    parents = np.array(
        [order.index(parent_of[u]) if u in parent_of else -1 for u in order]
    )

    n_bones = len(order)
    rest_t = np.zeros((n_bones, 3))
    rest_r = np.zeros((n_bones, 3))
    pre_r = np.zeros((n_bones, 3))
    post_r = np.zeros((n_bones, 3))
    rot_order = []
    for j, uid in enumerate(order):
        props = _properties70(models[uid])
        rest_t[j] = props.get("Lcl Translation", np.zeros(3))
        rest_r[j] = props.get("Lcl Rotation", np.zeros(3))
        pre_r[j] = props.get("PreRotation", np.zeros(3))
        post_r[j] = props.get("PostRotation", np.zeros(3))
        rot_order.append(ROTATION_ORDERS[int(props.get("RotationOrder", 0))])

    # Curve node -> model channel, and curve -> curve node axis.
    channel_of = {}
    for kind, child, parent, *rest in connections:
        if (
            kind == "OP"
            and parent in models
            and objects[child].name == "AnimationCurveNode"
        ):
            channel_of[child] = (parent, rest[0])
    curves = {}
    for kind, child, parent, *rest in connections:
        if kind == "OP" and parent in channel_of:
            model, channel = channel_of[parent]
            axis = "XYZ".index(rest[0][-1])
            curves[(model, channel, axis)] = _curve(objects[child])
    if not curves:
        raise ValueError(f"{path} has no skeletal animation curves")

    # Frame times: the union of all key times.
    times = np.unique(np.concatenate([t for t, _ in curves.values()]))
    n_frames = len(times)
    translation = np.repeat(rest_t[None], n_frames, axis=0)
    rotation = np.repeat(rest_r[None], n_frames, axis=0)
    for j, uid in enumerate(order):
        for channel, arr in (
            ("Lcl Translation", translation),
            ("Lcl Rotation", rotation),
        ):
            for axis in range(3):
                curve = curves.get((uid, channel, axis))
                if curve is not None:
                    arr[:, j, axis] = np.interp(times, *curve)

    return FbxSkeleton(
        names=names,
        parents=parents,
        times=times,
        rest_translation=rest_t,
        rest_rotation=rest_r,
        translation=translation,
        rotation=rotation,
        pre_rotation=pre_r,
        post_rotation=post_r,
        rotation_order=rot_order,
        global_settings=_properties70(top["GlobalSettings"]),
    )
