"""Per-step torque vs foot force for the 21402699 spawn_pose PD-hold runs (verify-coop)."""
import json, sys
import numpy as np
D = '/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/repo-gpu-20260923/spawn_pose/'
files = ['TaskV2-BHL-CubeToShelf-Blind-v0__pd_hold__pinch.json',
         'TaskV2-BHL-CubeToShelf-Blind-v0__pd_hold__standing.json',
         'TaskV2-BHL-CubeToShelf-Blind-v0__pd_hold__standing_z-0.03.json']
nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 25
out = {}
for fn in files:
    d = json.load(open(D + fn))
    ts = d['timeseries']
    print('=====', fn, '| init_z_offset', d['init_z_offset'], '|', d['verdict'])
    print(' step | sum foot force per env (A,B)            | max_computed (A)               | max_computed (B)               | applied A | ratio A | argmax A | tilt A (max) ')
    rows = []
    for t in ts[:nmax]:
        a, b = t['robot_a'], t['robot_b']
        ffa = [round(sum(f), 1) for f in a['foot_force']]
        ffb = [round(sum(f), 1) for f in b['foot_force']]
        ca = [round(x, 2) for x in a['max_computed_torque']]
        cb = [round(x, 2) for x in b['max_computed_torque']]
        print(' %3d | A%s B%s | %s | %s | %s | %s | %s | %.3f' % (
            t['step'], ffa, ffb, ca, cb, a['max_applied_torque'], a['max_torque_ratio'], a['argmax_applied_joint'], max(a['tilt_native'])))
        rows.append(dict(step=t['step'], ff_a=ffa, ff_b=ffb, comp_a=ca, comp_b=cb,
                         appl_a=a['max_applied_torque'], appl_b=b['max_applied_torque'],
                         arg_a=a['argmax_applied_joint'], arg_b=b['argmax_applied_joint'],
                         tilt_a=a['tilt_native'], tilt_b=b['tilt_native'], base_z_a=a['base_z']))
    out[fn] = rows
json.dump(out, open('/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/verify-coop/spawn_torque_table.json', 'w'), indent=1)
