"""Play a retargeted robot motion, with a frame counter and segment marking.

Keys in the viewer window:

    Space          pause or resume
    Left / Right   step one frame back or forward (pauses playback)
    Up / Down      jump one second forward or back
    Home           mark the current frame as the start of a segment
    End            mark the current frame as the end of the segment

Letter keys are left alone because MuJoCo's viewer already uses them to
toggle its visualisation options. Each segment closed with End is printed with its frames and times, and on exit
the script prints a robot_motion_to_csv.py command that exports every marked
segment. Frame numbers count from 0 in the file being played, so mark segments
on the full-take pickle (or its full CSV) for the times to match the export.
"""

import argparse
import os
import threading

import numpy as np

from general_motion_retargeting import RobotMotionViewer, load_robot_motion

# GLFW key codes passed to the viewer's key callback.
KEY_SPACE, KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 32, 262, 263, 264, 265
KEY_HOME, KEY_END = 268, 269


class Playback:
    """Frame index, pause state and marked segments, driven by key presses."""

    def __init__(self, n_frames, fps):
        self.n_frames = n_frames
        self.fps = fps
        self.frame = 0
        self.paused = False
        self.mark_in = None
        self.segments = []
        self.lock = threading.Lock()

    def _seek(self, delta):
        self.frame = int(np.clip(self.frame + delta, 0, self.n_frames - 1))

    def on_key(self, key):
        with self.lock:
            if key == KEY_SPACE:
                self.paused = not self.paused
            elif key in (KEY_LEFT, KEY_RIGHT):
                self.paused = True
                self._seek(1 if key == KEY_RIGHT else -1)
            elif key in (KEY_UP, KEY_DOWN):
                self._seek(round(self.fps) * (1 if key == KEY_UP else -1))
            elif key == KEY_HOME:
                self.mark_in = self.frame
                print(f"Segment start: frame {self.frame} ({self.time(self.frame)} s)")
            elif key == KEY_END:
                self._close_segment()

    def _close_segment(self):
        if self.mark_in is None:
            print("Press Home to mark a segment start before End.")
            return
        first, last = sorted((self.mark_in, self.frame))
        if first == last:
            print("A segment needs at least two frames.")
            return
        self.segments.append((first, last))
        self.mark_in = None
        print(
            f"Segment {len(self.segments)}: frames {first}-{last} "
            f"({self.time(first)}-{self.time(last)} s, {last - first + 1} frames)"
        )

    def time(self, frame):
        return f"{frame / self.fps:.2f}"

    def advance(self):
        """Return the frame to show now, then move on unless paused."""
        with self.lock:
            frame = self.frame
            if not self.paused:
                self.frame = (self.frame + 1) % self.n_frames
            return frame

    def overlay(self, frame):
        last = self.n_frames - 1
        lines = [
            f"Frame {frame} / {last}",
            f"Time  {self.time(frame)} / {self.time(last)} s",
        ]
        if self.paused:
            lines.append("Paused")
        if self.mark_in is not None:
            lines.append(f"Start marked at frame {self.mark_in}")
        for i, (first, last_frame) in enumerate(self.segments, 1):
            lines.append(f"Segment {i}: {first}-{last_frame}")
        return "\n".join(lines)

    def export_command(self, motion_path, robot):
        # Four decimals so that time * fps rounds back to the marked frame.
        segments = " ".join(
            f"--segment {first / self.fps:.4f} {last / self.fps:.4f}"
            for first, last in self.segments
        )
        stem = os.path.splitext(motion_path)[0]
        return (
            f"python scripts/robot_motion_to_csv.py --motion_file {stem}.pkl "
            f"--robot {robot} --csv_path {stem}.csv {segments}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=__doc__.split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    parser.add_argument(
        "--no_frame_counter",
        action="store_true",
        help="Do not draw the frame counter in the viewer window.",
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

    playback = Playback(len(motion_root_pos), motion_fps)

    env = RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=motion_fps,
        camera_follow=False,
        record_video=args.record_video,
        video_path=args.video_path,
        keyboard_callback=playback.on_key,
    )

    try:
        while env.viewer.is_running():
            frame_idx = playback.advance()
            env.step(
                motion_root_pos[frame_idx],
                motion_root_rot[frame_idx],
                motion_dof_pos[frame_idx],
                rate_limit=True,
                overlay_text=None
                if args.no_frame_counter
                else playback.overlay(frame_idx),
            )
    except KeyboardInterrupt:
        pass
    finally:
        if playback.segments:
            print("Export the marked segments with:")
            print(playback.export_command(robot_motion_path, robot_type))
        env.close()
