"""Joint exceedance of the predeclared ResumeBound by truth on the calib.py windows."""
import math, sys
import numpy as np
A = np.load(sys.argv[1]); T, L, rot, pe, ye, z = A.T
ok = (pe <= 0.15 + 0.6 * L) & (ye <= math.radians(10) + 0.4 * rot)
print('joint violation all windows: %.3f%% (n=%d); nonzero-command windows: %.3f%%; T<=3s: %.3f%%; zero-command: %.3f%%' % (
    100 * (~ok).mean(), len(ok), 100 * (~ok[z == 0]).mean(), 100 * (~ok[T <= 3]).mean(), 100 * (~ok[z == 1]).mean()))
