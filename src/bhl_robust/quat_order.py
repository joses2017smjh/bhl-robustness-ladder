"""Which order the running Isaac Lab stores quaternions in, and a converter into it.

Isaac Lab 2.3.2 (`BHL_STACK=v51`) takes `(w, x, y, z)`. Isaac Lab 3.0
(`BHL_STACK=v60`) takes `(x, y, z, w)` -- in every `OffsetCfg.rot`, every
`InitialStateCfg.rot`, and in `matrix_from_quat` itself, which unpacks
`i, j, k, r` there and `r, i, j, k` here. Both stacks accept any 4-tuple
silently, so a literal written for one is a different rotation on the other,
and nothing errors.

Every literal in this repo is written `(w, x, y, z)`, because every published
number before the v60 stack came from 2.3.2. On 3.0 the camera pose
`(0.9848, 0, 0.1736, 0)` -- 20 degrees of down-pitch -- is read as a half-turn
about an axis 10 degrees off +x: the camera looks 20 degrees *up* and the image
is upside down. That is what the maze stereo pair did for every run on v60.

So a pose literal goes through `native_quat` at the point it enters a config.
The order is probed from Isaac Lab rather than keyed off a version string, so a
release that changes it again cannot slip past; on v51 the conversion is the
identity and no v51 number moves.

Import-safe on the login node: Isaac Lab is only imported when the order is
first asked for, which happens inside a config factory, after the simulator has
started.
"""

from __future__ import annotations

from functools import lru_cache

WXYZ = "wxyz"
XYZW = "xyzw"


@lru_cache(maxsize=1)
def quat_order() -> str:
    """`"xyzw"` or `"wxyz"`, asked of the Isaac Lab in this interpreter.

    `(0, 0, 0, 1)` is the identity only if it is stored `(x, y, z, w)`; stored
    `(w, x, y, z)` it is a half-turn about z, whose matrix has -1 in the corner.
    """
    import torch
    from isaaclab.utils.math import matrix_from_quat

    m = matrix_from_quat(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    return XYZW if float(m[0, 0, 0]) > 0.5 else WXYZ


def reorder(wxyz, order: str) -> tuple[float, float, float, float]:
    """A `(w, x, y, z)` quaternion laid out in `order`."""
    if order not in (WXYZ, XYZW):
        raise ValueError(f"unknown quaternion order {order!r}")
    w, x, y, z = (float(v) for v in wxyz)
    return (x, y, z, w) if order == XYZW else (w, x, y, z)


def native_quat(wxyz) -> tuple[float, float, float, float]:
    """A `(w, x, y, z)` literal in the order this Isaac Lab will read it."""
    return reorder(wxyz, quat_order())
