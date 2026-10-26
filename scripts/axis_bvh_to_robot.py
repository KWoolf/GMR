import argparse
import os
import pathlib
import pickle
import time

import numpy as np
from rich import print
from tqdm import tqdm

from general_motion_retargeting import IK_CONFIG_DICT, ROBOT_XML_DICT
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.axis_studio import load_axis_bvh_file


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
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Retarget a BVH file exported from Noitom Axis Studio to a robot."
    )
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file exported from Axis Studio.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--robot",
        choices=sorted(IK_CONFIG_DICT["bvh_axis"]),
        default="unitree_r1",
    )
    parser.add_argument(
        "--motion_fps",
        default=50,
        type=float,
        help="Frame rate to resample the capture to before retargeting.",
    )
    parser.add_argument(
        "--source_fps",
        default=None,
        type=float,
        help="Capture rate of the BVH file. Defaults to the Perception Neuron "
        "rate nearest the file's (rounded) Frame Time.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Retarget without opening the interactive viewer.",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion in the viewer.",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    parser.add_argument(
        "--video_path",
        default=None,
        help="Render the retargeted motion offscreen to this video file.",
    )
    args = parser.parse_args()

    frames, actual_human_height, motion_fps = load_axis_bvh_file(
        args.bvh_file, target_fps=args.motion_fps, source_fps=args.source_fps
    )
    print(
        f"Loaded {len(frames)} frames at {motion_fps} fps, "
        f"performer height {actual_human_height:.3f} m"
    )

    retargeter = GMR(
        src_human="bvh_axis",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    # Start the solver at the performer's heading on the first frame. From the
    # default pose facing +X, a take that starts facing elsewhere can leave the
    # IK in a local minimum with the pelvis pitched over.
    hips_pos, hips_quat = frames[0]["Hips"]
    q_init = retargeter.configuration.data.qpos.copy()
    q_init[:2] = hips_pos[:2] * retargeter.human_scale_table["Hips"]
    q_init[3:7] = hips_quat
    retargeter.configuration.update(q_init)
    for _ in range(20):
        retargeter.retarget(frames[0])

    robot_motion_viewer = None
    if not args.headless:
        from general_motion_retargeting import RobotMotionViewer

        robot_motion_viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=motion_fps,
            transparent_robot=0,
        )

    qpos_list = []
    pbar = tqdm(total=len(frames), desc="Retargeting")
    i = 0
    start_time = time.time()
    while True:
        pbar.update(1)
        qpos = retargeter.retarget(frames[i])
        if robot_motion_viewer is not None:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )
        if len(qpos_list) < len(frames):  # keep only the first pass when looping
            qpos_list.append(qpos)

        if args.loop and robot_motion_viewer is not None:
            i = (i + 1) % len(frames)
        else:
            i += 1
            if i >= len(frames):
                break
    pbar.close()
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

    if args.video_path is not None:
        render_video(args.robot, qpos_list, motion_fps, args.video_path)
