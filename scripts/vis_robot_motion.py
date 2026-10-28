import argparse
import os

import numpy as np

from general_motion_retargeting import RobotMotionViewer, load_robot_motion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")

    parser.add_argument(
        "--robot_motion_path",
        type=str,
        required=True,
        help="GMR pickle, or a CSV from scripts/robot_motion_to_csv.py.",
    )
    parser.add_argument(
        "--csv_fps",
        type=float,
        default=50.0,
        help="Frame rate of a CSV motion file (CSV files do not store it).",
    )

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/example.mp4")

    args = parser.parse_args()

    robot_type = args.robot
    robot_motion_path = args.robot_motion_path

    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")

    if robot_motion_path.endswith(".csv"):
        # Rows are root position, root quaternion (xyzw) and joint positions.
        motion = np.loadtxt(robot_motion_path, delimiter=",", ndmin=2)
        motion_fps = args.csv_fps
        motion_root_pos = motion[:, :3]
        motion_root_rot = motion[:, [6, 3, 4, 5]]  # from xyzw to wxyz
        motion_dof_pos = motion[:, 7:]
    else:
        (
            motion_data,
            motion_fps,
            motion_root_pos,
            motion_root_rot,
            motion_dof_pos,
            motion_local_body_pos,
            motion_link_body_list,
        ) = load_robot_motion(robot_motion_path)

    env = RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=motion_fps,
        camera_follow=False,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    frame_idx = 0
    while True:
        env.step(
            motion_root_pos[frame_idx],
            motion_root_rot[frame_idx],
            motion_dof_pos[frame_idx],
            rate_limit=True,
        )
        frame_idx += 1
        if frame_idx >= len(motion_root_pos):
            frame_idx = 0
    env.close()
