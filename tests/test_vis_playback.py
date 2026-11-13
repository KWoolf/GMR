"""Playback controls used by scripts/vis_robot_motion.py, without a viewer."""

from general_motion_retargeting import playback as vis


def test_pause_step_and_segments():
    pb = vis.Playback(n_frames=200, fps=50)
    assert [pb.advance() for _ in range(3)] == [0, 1, 2]
    pb.on_key(vis.KEY_HOME)  # start at frame 3
    pb.on_key(vis.KEY_UP)  # +1 s
    assert pb.frame == 53
    pb.on_key(vis.KEY_RIGHT)  # pauses and steps
    assert pb.paused and pb.frame == 54
    assert pb.advance() == 54 and pb.advance() == 54
    pb.on_key(vis.KEY_END)
    assert pb.segments == [(3, 54)]
    assert "--segment 0.0600 1.0800" in pb.export_command("take.pkl", "unitree_r1")
    pb.on_key(vis.KEY_DOWN)
    pb.on_key(vis.KEY_DOWN)
    assert pb.frame == 0  # clamped at the first frame
