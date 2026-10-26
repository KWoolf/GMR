# Unitree R1 (26 DoF)

`r1_26dof.urdf` and `meshes/` are Unitree's description of the R1 with 6 joints per leg, 2 waist joints (roll, yaw), 5 per arm and 2 head joints (pitch, yaw). The ankle parallel-linkage parts are fixed links in the URDF and are merged into their parents in the MJCF.

`r1_mocap_26dof.xml` is generated from the URDF:

```bash
python assets/unitree_r1/build_r1_mjcf.py
```

## Axis Studio IK config

`general_motion_retargeting/ik_configs/bvh_axis_to_r1.json` maps the Axis Studio skeleton to the R1. The Axis Studio loader returns every bone with the identity orientation in the rest T-pose, so each rotation offset in the config is the orientation of the robot link when the robot stands in the matching T-pose. Legs, pelvis, torso and head links are world-aligned at the zero pose, so their offsets are the identity. The arm offsets come from this T-pose, with the arm reaching sideways and the elbow as straight as the R1 geometry allows (about 32 degrees short of straight between joint origins):

| joint | left | right |
| --- | --- | --- |
| shoulder_pitch | 0.5844 | 0.5844 |
| shoulder_roll | 1.5452 | -1.5452 |
| shoulder_yaw | 0.5666 | -0.5666 |
| elbow | 1.2739 | 1.2739 |

The scale table is the ratio of robot to human distances from the pelvis for each matched body, for a 1.73 m performer; other performers are scaled by their stature, which the loader reads from the BVH rest skeleton.
