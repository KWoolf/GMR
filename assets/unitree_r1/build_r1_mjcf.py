"""Build the GMR MJCF for the Unitree R1 (26 DoF) from Unitree's URDF.

MuJoCo's URDF importer drops the floating base and turns the root link into
the world body, so this script wraps the root link's geoms and children in a
`pelvis` body with a free joint and restores the root link's inertia.

Usage (from the repository root):
    python assets/unitree_r1/build_r1_mjcf.py
"""

import pathlib
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).parent
URDF = HERE / "r1_26dof.urdf"
OUT = HERE / "r1_mocap_26dof.xml"
ROOT_LINK = "pelvis_link"
ROOT_HEIGHT = 0.76


def root_inertial(urdf_path):
    link = ET.parse(urdf_path).getroot().find(f"link[@name='{ROOT_LINK}']")
    inertial = link.find("inertial")
    pos = inertial.find("origin").get("xyz")
    mass = inertial.find("mass").get("value")
    i = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
    full = f"{i['ixx']} {i['iyy']} {i['izz']} {i['ixy']} {i['ixz']} {i['iyz']}"
    return {"pos": pos, "mass": mass, "fullinertia": full}


def main():
    # The URDF references meshes as "meshes/<file>" while also setting
    # meshdir="meshes", so strip the prefix before handing it to MuJoCo.
    urdf_text = URDF.read_text().replace('filename="meshes/', 'filename="')
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", dir=HERE, delete=False) as f:
        f.write(urdf_text)
        tmp_urdf = pathlib.Path(f.name)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_urdf))
        with tempfile.TemporaryDirectory() as d:
            raw = pathlib.Path(d) / "raw.xml"
            mujoco.mj_saveLastXML(str(raw), model)
            tree = ET.parse(raw)
    finally:
        tmp_urdf.unlink()

    root = tree.getroot()
    root.set("model", "r1_mocap")
    root.find("compiler").set("meshdir", "meshes")
    worldbody = root.find("worldbody")

    pelvis = ET.Element("body", {"name": "pelvis", "pos": f"0 0 {ROOT_HEIGHT}"})
    ET.SubElement(pelvis, "inertial", root_inertial(URDF))
    ET.SubElement(pelvis, "freejoint", {"name": "floating_base_joint"})
    for child in list(worldbody):
        worldbody.remove(child)
        pelvis.append(child)
    worldbody.append(pelvis)

    ET.indent(tree, space="  ")
    tree.write(OUT)

    check = mujoco.MjModel.from_xml_path(str(OUT))
    hinge = [check.joint(j).name for j in range(check.njnt) if check.jnt_type[j] == 3]
    assert len(hinge) == 26, hinge
    assert np.isclose(check.body("pelvis").mass[0], float(root_inertial(URDF)["mass"]))
    print(f"Wrote {OUT} with {len(hinge)} hinge joints")


if __name__ == "__main__":
    main()
