"""Playback state for the robot motion viewer: frame index, pause and segments.

Driven by key presses from MuJoCo's passive viewer (GLFW key codes), and used
by scripts/vis_robot_motion.py.
"""

import os
import threading

import numpy as np

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
