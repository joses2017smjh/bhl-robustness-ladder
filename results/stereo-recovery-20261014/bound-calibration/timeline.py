import json, sys, glob, math, os
import numpy as np
root = sys.argv[1]
for ep in sorted(glob.glob(os.path.join(root, '*/bounded_latest-*'))):
    name = os.path.basename(ep)
    frames = [json.loads(l) for l in open(os.path.join(ep, 'native/native_frames.jsonl'))]
    cam = [f for f in frames if f.get('new_native_camera_frame')]
    episode = json.load(open(os.path.join(ep, 'episode.json')))
    log = open(os.path.join(ep, 'native/orb/native-orb.log')).read()
    events = [l for l in log.splitlines() if any(k in l for k in ('Relocalized', 'Creation of new map with id', 'Active map', 'New Map created', 'Fail to track', 'Reseting', 'A new map', 'Track Lost'))]
    # timeline of states
    seq = []
    prev = None
    for f in cam:
        key = (f['native_ORB_tracking_state'], f['native_ORB_map_id'], f['tracked'], f['state'])
        if key != prev:
            seq.append((round(f['timestamp_s'], 2),) + key)
            prev = key
    trace = episode['trace']
    xy = np.array([t['evaluator_xy'] for t in trace]); ts = np.array([t['timestamp_s'] for t in trace])
    print('=' * 100)
    print(name, 'success', episode['success'], 'final dist %.2f' % trace[-1]['evaluator_goal_distance_m'], 'min dist %.2f' % min(t['evaluator_goal_distance_m'] for t in trace))
    print(' invalidations:', [(round(e['timestamp_s'], 2), e['reasons']) for e in json.load(open(os.path.join(ep, 'native/stream_receipt.json')))['coordinate_epochs']['events']])
    print(' log events:', ' | '.join(e.strip()[:40] for e in events))
    for s in seq[:40]:
        print('   t=%6.2f state=%s map=%s tracked=%s pkt=%s' % s)
