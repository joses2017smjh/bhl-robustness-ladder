"""verify-turning: checkpoint structure, std values, numpy-actor vs ONNX vs TorchScript agreement (read-only)."""
import glob, json, sys
import numpy as np
import torch
import onnxruntime as ort
sys.path.insert(0, "/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/turning")
import mj_probe2 as P

torch.set_num_threads(1)
runs = ["arms-turn-turncmd-s0", "arms-turn-turncmd-s1", "arms-turn-turnboth-s1", "arms-turn-turnhip-s0",
        "arms-turn-turnboth-s0", "arms-dr1.0-s0"]
rng = np.random.default_rng(7)
out = {}
for r in runs:
    ck = P.ckpt_path(r, None)
    d = ck.rsplit("/", 1)[0]
    raw = torch.load(ck, map_location="cpu", weights_only=False)
    sd = raw["model_state_dict"]
    keys = list(sd.keys())
    nonmlp = [k for k in keys if not (k.startswith("actor.") or k.startswith("critic."))]
    std = sd["std"].numpy()
    act = P.NpActor(ck, 0.0, 0, "all")
    sess = ort.InferenceSession(d + "/exported/policy.onnx")
    iname = sess.get_inputs()[0].name
    ts = torch.jit.load(d + "/exported/policy.pt", map_location="cpu").eval()
    # observations: plausible scale per term (cmd, angvel, gravity, qpos, qvel, last action)
    N = 2000
    obs = np.concatenate([rng.uniform(-1, 1, (N, 3)), rng.normal(0, 0.5, (N, 3)),
                          np.tile([0, 0, -1.0], (N, 1)) + rng.normal(0, 0.05, (N, 3)),
                          rng.normal(0, 0.3, (N, 22)), rng.normal(0, 2.0, (N, 22)), rng.normal(0, 3.0, (N, 22))], 1).astype(np.float32)
    a_np = np.stack([act.mean(o) for o in obs])
    a_ox = np.concatenate([sess.run(None, {iname: obs[i:i + 1]})[0] for i in range(N)])
    with torch.inference_mode():
        a_ts = ts(torch.from_numpy(obs)).numpy()
    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))
    out[r] = {"ckpt": ck, "iter": raw.get("iter"), "nonmlp_keys": nonmlp,
              "std_arms_mean": round(float(std[:10].mean()), 3), "std_legs_mean": round(float(std[10:].mean()), 3),
              "std_legs_minmax": [round(float(std[10:].min()), 3), round(float(std[10:].max()), 3)],
              "std_arms_minmax": [round(float(std[:10].min()), 3), round(float(std[:10].max()), 3)],
              "np_vs_onnx_maxabs": float(np.abs(a_np - a_ox).max()), "np_vs_onnx_rel": rel(a_np, a_ox),
              "np_vs_ts_maxabs": float(np.abs(a_np - a_ts).max()), "onnx_input": iname}
    print(r, json.dumps(out[r]), flush=True)
json.dump(out, open("/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/verify-turning/ckpt_check.json", "w"), indent=1)
