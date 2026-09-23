"""Mission 7 result artifacts stay small enough to live in git history.

A name-by-name ignore list stopped covering the route probes when they began
writing ``routes.json``, ``doors-<n>.json`` and whole episode rows into
``result.json``.  One ``result.json`` reached 84 MB and one ``episodes.json``
951 MB, and 1.4 GB sat untracked-but-unignored under ``results/`` -- a single
``git add -A`` away from a history that cannot be undone without a rewrite.

Patterns alone cannot close this: ``result.json`` is both the compact verdict
the docs cite and, historically, the raw trace.  So this test asserts the
invariant the patterns are trying to express -- nothing git would commit under
``results/mission7-*`` is large -- and fails when a new oversized artifact
appears, rather than letting it be committed and discovered later.

The check is deliberately over what git *would* commit (tracked plus
untracked-not-ignored), never a filesystem walk: the ignored raw traces are
gigabytes and are supposed to be.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

#: Largest artifact git may commit under results/mission7-*.  The largest
#: compact artifact actually tracked today is 139 KB, so this is generous by
#: nearly two orders of magnitude.  Raise it only with a reason; the correct
#: response to a failure is to make the producer write a summary, or to ignore
#: the raw trace by shape in .gitignore.
MAX_ARTIFACT_BYTES = 5 * 1024 * 1024

_SCOPE = "results/mission7-*"


def _git(*args):
    return subprocess.run(("git", "-C", str(_REPO)) + args,
                          capture_output=True, text=True, check=True).stdout.split("\0")


def committable_paths():
    """Paths under the Mission 7 results scope that git would include."""
    tracked = _git("ls-files", "-z", "--", _SCOPE)
    untracked = _git("ls-files", "-z", "--others", "--exclude-standard", "--", _SCOPE)
    return sorted({name for name in tracked + untracked if name})


class ResultArtifactSize(unittest.TestCase):
    def setUp(self):
        if shutil.which("git") is None or not (_REPO / ".git").exists():
            self.skipTest("not a git checkout")

    def test_no_oversized_committable_artifact(self):
        oversized = []
        for name in committable_paths():
            path = _REPO / name
            # A path can be listed and already gone; that is not this test's problem.
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size > MAX_ARTIFACT_BYTES:
                oversized.append((size, name))
        oversized.sort(reverse=True)
        detail = "\n".join(f"  {size / 1048576:8.2f} MB  {name}"
                           for size, name in oversized)
        self.assertEqual(
            oversized, [],
            f"\n{len(oversized)} Mission 7 artifact(s) over "
            f"{MAX_ARTIFACT_BYTES / 1048576:.0f} MB would be committed:\n{detail}\n"
            "Write a compact summary beside the trace and ignore the trace by "
            "shape in .gitignore, rather than raising the limit.")


if __name__ == "__main__":
    unittest.main()
