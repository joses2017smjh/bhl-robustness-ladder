"""Ranges across all 8 robot-envs (2 robots x 4 envs) per step: foot force, computed torque, applied ratio, tilt."""
import json
D = '/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/repo-gpu-20260923/spawn_pose/'
res = {}
for tag in ['pinch', 'standing', 'standing_z-0.03']:
    d = json.load(open(D + f'TaskV2-BHL-CubeToShelf-Blind-v0__pd_hold__{tag}.json'))
    ts = d['timeseries']
    print('=====', tag, 'init_z_offset', d['init_z_offset'], d['verdict'])
    rows = []
    for t in ts:
        ff, comp, ratio, tilt, arg = [], [], [], [], []
        for r in ('robot_a', 'robot_b'):
            x = t[r]
            ff += [sum(f) for f in x['foot_force']]
            comp += x['max_computed_torque']; ratio += x['max_torque_ratio']; tilt += x['tilt_native']
            arg += x['argmax_applied_joint']
        rows.append(dict(step=t['step'], ff_min=min(ff), ff_max=max(ff), comp_min=min(comp), comp_max=max(comp),
                         n_sat=sum(1 for v in ratio if v >= 0.999), tilt_max=max(tilt), tilt_min=min(tilt), args=sorted(set(int(a) for a in arg))))
    for row in rows[:12]:
        print('  step %3d  footF %6.1f-%6.1f N  comp %5.2f-%5.2f Nm  saturated %d/8  tilt %.3f-%.3f  argmax %s' % (
            row['step'], row['ff_min'], row['ff_max'], row['comp_min'], row['comp_max'], row['n_sat'], row['tilt_min'], row['tilt_max'], row['args']))
    # steps where any computed torque >= 27 Nm
    big = [(r['step'], round(r['comp_max'], 2), round(r['tilt_max'], 3)) for r in rows if r['comp_max'] >= 27.0]
    print('  steps with any computed >= 27 Nm:', big[:12], '... total', len(big))
    # first step with computed > 6 on any env, with tilt
    first6 = next((r for r in rows[1:] if r['comp_max'] > 6.0), None)
    print('  first step>=1 with computed > 6 Nm on any robot-env:', first6['step'], round(first6['comp_max'], 2), 'tilt_max', round(first6['tilt_max'], 3))
    s17 = [r for r in rows[1:8]]
    print('  steps 1-7 computed range all robot-envs: %.2f-%.2f' % (min(r['comp_min'] for r in s17), max(r['comp_max'] for r in s17)))
    s25 = [r for r in rows[2:6]]
    print('  steps 2-5 computed range all robot-envs: %.2f-%.2f; tilt max %.3f; footF %.0f-%.0f' % (min(r['comp_min'] for r in s25), max(r['comp_max'] for r in s25), max(r['tilt_max'] for r in s25), min(r['ff_min'] for r in s25), max(r['ff_max'] for r in s25)))
    res[tag] = rows
json.dump(res, open('/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/verify-coop/spawn_ranges.json', 'w'), indent=0)
