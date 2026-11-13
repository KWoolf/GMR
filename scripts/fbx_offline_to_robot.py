"""Retarget an offline FBX motion to a robot.

Two inputs are accepted:

- A .fbx file exported by Organic Motion (the "om_fbx" source), read directly
  with GMR's own FBX reader.
- A .pkl file of per-frame OptiTrack FBX data (the "fbx_offline" source),
  as produced by an external FBX SDK conversion.
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
from rich import print
from tqdm import tqdm

from general_motion_retargeting import IK_CONFIG_DICT, ROBOT_XML_DICT
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.om_fbx import load_om_fbx_file

DEFAULT_ROBOT = {"om_fbx": "unitree_r1", "fbx_offline": "unitree_g1"}


def load_optitrack_fbx_motion_file(motion_file):
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
    return motion_data


def offset_to_ground(retargeter: GMR, motion_data):
    offset = np.inf
    for human_data in motion_data:
        human_data = retargeter.to_numpy(human_data)
        human_data = retargeter.scale_human_data(
            human_data, retargeter.human_root_name, retargeter.human_scale_table
        )
        human_data = retargeter.offset_human_data(
            human_data, retargeter.pos_offsets1, retargeter.rot_offsets1
        )
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            if pos[2] < offset:
                offset = pos[2]

    return offset


def render_video(robot, qpos_list, fps, video_path, width=640, height=480):
    """Render the retargeted motion offscreen with a camera that follows the pelvis."""
    import imageio
    import mujoco as mj

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=height, width=width)
    camera = mj.MjvCamera()
    camera.distance = 2.5
    camera.elevation = -10
    camera.azimuth = 150
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    writer = imageio.get_writer(video_path, fps=fps)
    for qpos in qpos_list:
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        camera.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, camera)
        writer.append_data(renderer.render())
    writer.close()
    print(f"Saved video to {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--motion_file",
        help="Organic Motion .fbx file, or OptiTrack FBX data converted to .pkl.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--robot",
        choices=sorted(
            set(IK_CONFIG_DICT["fbx_offline"]) | set(IK_CONFIG_DICT["om_fbx"])
        ),
        default=None,
        help="Target robot. Defaults to unitree_r1 for .fbx and unitree_g1 for .pkl.",
    )
    parser.add_argument(
        "--motion_fps",
        default=None,
        type=float,
        help="Resample an .fbx capture to this frame rate (e.g. 50 for mjlab). "
        "Defaults to the capture rate.",
    )
    parser.add_argument(
        "--start",
        default=None,
        type=float,
        help="Start of the .fbx segment to retarget, in seconds.",
    )
    parser.add_argument(
        "--end",
        default=None,
        type=float,
        help="End of the .fbx segment to retarget, in seconds.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Retarget without opening the interactive viewer.",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
        help="Record the interactive viewer to --video_path.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Video file. Rendered offscreen when --record_video is not set.",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    args = parser.parse_args()

    if args.motion_file.lower().endswith(".fbx"):
        src_human = "om_fbx"
        print(f"Loading Organic Motion FBX file: {args.motion_file}")
        data_frames, human_height, motion_fps = load_om_fbx_file(
            args.motion_file,
            target_fps=args.motion_fps,
            start=args.start,
            end=args.end,
        )
        print(f"Performer height {human_height:.3f} m")
    else:
        src_human = "fbx_offline"
        print(f"Loading OptiTrack FBX motion file: {args.motion_file}")
        data_frames = load_optitrack_fbx_motion_file(args.motion_file)
        motion_fps = 120
        human_height = None
    print(f"Loaded {len(data_frames)} frames at {motion_fps} fps")

    robot = args.robot or DEFAULT_ROBOT[src_human]
    if robot not in IK_CONFIG_DICT[src_human]:
        parser.error(
            f"No {src_human} IK config for {robot}; "
            f"choose from {sorted(IK_CONFIG_DICT[src_human])}"
        )

    if src_human == "om_fbx":
        # GMR multiplies the config's scale table by actual / assumed height,
        # which shrinks a shorter performer a second time. Passing
        # assumed^2 / actual makes that factor assumed / actual instead, so the
        # scaled motion is the same size whatever the performer's stature.
        with open(IK_CONFIG_DICT[src_human][robot]) as f:
            assumed_height = json.load(f)["human_height_assumption"]
        retargeter = GMR(
            src_human=src_human,
            tgt_robot=robot,
            actual_human_height=assumed_height**2 / human_height,
        )
    else:
        retargeter = GMR(src_human=src_human, tgt_robot=robot, actual_human_height=1.8)

    height_offset = offset_to_ground(retargeter, data_frames)
    retargeter.set_ground_offset(height_offset)

    if src_human == "om_fbx":
        # Start the solver at the performer's heading on the first frame. From
        # the default pose facing +X, a take that starts facing elsewhere can
        # leave the IK in a local minimum with the pelvis pitched over.
        hips_pos, hips_quat = data_frames[0]["Hips"]
        q_init = retargeter.configuration.data.qpos.copy()
        q_init[:2] = hips_pos[:2] * retargeter.human_scale_table["Hips"]
        q_init[3:7] = hips_quat
        retargeter.configuration.update(q_init)
        for _ in range(20):
            retargeter.retarget(data_frames[0])

    robot_motion_viewer = None
    if not args.headless:
        from general_motion_retargeting import RobotMotionViewer

        robot_motion_viewer = RobotMotionViewer(
            robot_type=robot,
            motion_fps=motion_fps,
            transparent_robot=1,
            record_video=args.record_video,
            video_path=args.video_path or "videos/fbx_example.mp4",
            camera_follow=False,
        )

    qpos_list = []
    start_time = time.time()
    for human_data in tqdm(data_frames, desc="Retargeting"):
        qpos = retargeter.retarget(human_data)
        qpos_list.append(qpos)
        if robot_motion_viewer is not None:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
            )
    print(f"Retargeted {len(qpos_list)} frames in {time.time() - start_time:.1f} s")

    if robot_motion_viewer is not None:
        robot_motion_viewer.close()

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_array = np.array(qpos_list)
        motion_data = {
            "fps": motion_fps,
            "root_pos": qpos_array[:, :3],
            # save from wxyz to xyzw
            "root_rot": qpos_array[:, [4, 5, 6, 3]],
            "dof_pos": qpos_array[:, 7:],
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    if args.video_path is not None and not args.record_video:
        render_video(robot, qpos_list, motion_fps, args.video_path)
