"""Robustness overlays for the Berkeley Humanoid Lite locomotion tasks.

Upstream BHL is pinned as a submodule and left unmodified. Everything here
subclasses its configs and registers new gym task ids alongside the originals,
so upstream stays cleanly re-pinnable.

Note: nothing in this package may be imported before Isaac Sim's SimulationApp
is instantiated. Carbonite aborts on any Omniverse import that precedes it,
and the upstream configs pull in isaaclab at module scope.
"""

__version__ = "0.1.0"
