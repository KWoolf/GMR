"""BVH parsing tests that do not need a robot model."""

import numpy as np

from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

# Axis Neuron can leave a blank line between MOTION and Frames.
BVH_WITH_BLANK_LINE = """HIERARCHY
ROOT Hips
{
    OFFSET 0.000 90.000 0.000
    CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
    JOINT Spine
    {
        OFFSET 0.000 10.000 0.000
        CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
        End Site
        {
            OFFSET 0.000 10.000 0.000
        }
    }
}
MOTION

Frames: 2
Frame Time: 0.008
0 90 0 0 0 0 0 10 0 0 0 0
1 90 0 10 0 0 0 10 0 0 0 0
"""


def test_read_bvh_skips_blank_lines(tmp_path):
    path = tmp_path / "blank_line.bvh"
    path.write_text(BVH_WITH_BLANK_LINE.replace("\n", "\r\n"), newline="")
    data = read_bvh(str(path))
    assert data.pos.shape == (2, 2, 3)
    np.testing.assert_allclose(data.pos[:, 0, 0], [0.0, 1.0])
