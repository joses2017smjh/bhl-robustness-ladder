import math

import numpy as np
import pytest

from bhl_robust.quat_order import WXYZ, XYZW, reorder, unpack_wxyz


def _yaw(angle):
    return (math.cos(angle/2), 0.0, 0.0, math.sin(angle/2))      # (w, x, y, z)


@pytest.mark.parametrize("order", [WXYZ, XYZW])
def test_unpack_recovers_wxyz_from_either_layout(order):
    wxyz = _yaw(math.pi/2)
    stored = np.array([reorder(wxyz, order), reorder(_yaw(-math.pi/2), order)])
    w, x, y, z = unpack_wxyz(stored, order)
    assert np.allclose(w, [wxyz[0], wxyz[0]]) and np.allclose(z, [wxyz[3], -wxyz[3]])
    # An upright, yawed body has R[2, 2] = 1 whatever the storage layout.
    up_z = 1.0 - 2.0*(x*x + y*y)
    assert np.allclose(up_z, 1.0)
    assert np.allclose(np.arccos(np.clip(up_z, -1, 1)), 0.0)


def test_misreading_xyzw_as_wxyz_is_the_spawn_bug():
    stored = np.array([reorder(_yaw(math.pi/2), XYZW)])
    w, x, y, z = unpack_wxyz(stored, WXYZ)          # the old, wrong read
    up_z = 1.0 - 2.0*(x*x + y*y)
    assert abs(float(np.arccos(np.clip(up_z, -1, 1))) - math.pi/2) < 1e-6


def test_unknown_order_rejected():
    with pytest.raises(ValueError):
        unpack_wxyz(np.zeros((1, 4)), "zyxw")
