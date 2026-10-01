import sys, json
import numpy as np, onnxruntime as ort
sys.path.insert(0, '/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/src')
from bhl_robust.navgym import env as ng
from PIL import Image, ImageDraw
SP = sys.argv[1]
REPO = '/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder'


def draw_maze(mz, boxes, S=60, pad=24):
    xmin, xmax, ymin, ymax = mz.bounds()
    W, H = int((xmax - xmin) * S) + 2 * pad, int((ymax - ymin) * S) + 2 * pad
    img = Image.new('RGB', (W, H), (245, 245, 245)); d = ImageDraw.Draw(img)
    px = lambda x, y: (pad + (x - xmin) * S, H - pad - (y - ymin) * S)
    for b in np.asarray(boxes):
        a, c = px(b[0], b[2]), px(b[1], b[3])          # boxes are (xmin, xmax, ymin, ymax)
        d.rectangle((min(a[0], c[0]), min(a[1], c[1]), max(a[0], c[0]), max(a[1], c[1])), fill=(40, 45, 60))
    return img, d, px


print('boxes sample', np.asarray(ng.wall_boxes(ng.generate(6, 6, 50009, extra_openings=1)))[:2], flush=True)
for k in [int(a) for a in sys.argv[2:]]:
    seed = 50000 + k
    tiles = []
    for actor in ('armV4-s5', 'armV4-s6'):
        env = ng.MazeNavEnv(sizes=((6, 6),), randomize_dynamics=False, seed_base=seed, seed_span=1, version=2, max_steps=4500)
        obs, _ = env.reset(seed=seed); env.yaw = 0.0; env._scan(); obs = env._obs()
        sess = ort.InferenceSession(f'{REPO}/results/navgym-v4-20260928/{actor}/actor.onnx'); keys = [i.name for i in sess.get_inputs()]
        path = [(env.x, env.y)]; info = {}
        for t in range(4500):
            a = np.clip(sess.run(None, {kk: np.asarray(obs[kk], np.float32)[None] for kk in keys})[0][0], -1, 1)
            obs, r, term, trunc, info = env.step(a); path.append((env.x, env.y))
            if term or trunc:
                break
        img, d, px = draw_maze(env.maze, env.boxes)
        d.line([px(*p) for p in path[::5]], fill=(255, 120, 0), width=2)
        pj = json.load(open(f'{REPO}/results/navgym-v4-transfer-20260930/{actor}/seed{seed}.json'))
        d.line([px(*t['xy']) for t in pj['trace']], fill=(0, 140, 255), width=2)
        sx, sy = env.maze.centre(env.maze.start); gx, gy = env.maze.centre(env.maze.goal)
        for (x, y), col in (((sx, sy), (0, 160, 0)), ((gx, gy), (200, 0, 0))):
            p = px(x, y); d.ellipse((p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7), fill=col)
        d.text((5, 4), f"{actor} maze {seed}: gym(orange) {info.get('outcome')} | physics(blue) {'goal' if pj['success'] else 'time-out'}", fill=(0, 0, 0))
        tiles.append(img)
        print(actor, seed, info.get('outcome'), len(path), flush=True)
    W = sum(t.size[0] for t in tiles); H = max(t.size[1] for t in tiles)
    canvas = Image.new('RGB', (W, H), 'white'); x = 0
    for t in tiles:
        canvas.paste(t, (x, 0)); x += t.size[0]
    canvas.save(f'{SP}/traj_{seed}.png'); print('saved', seed, flush=True)
