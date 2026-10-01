"""verify-turning: re-run ONE noise-free and ONE noisy episode for one non-turner (TurnBoth-s1, reset seed 0, wz +0.6).

mode 'probe0'  : investigator's mj_probe2.run, noise 0      (saved: exp1 yaw +13.7, lo_turn [0,0])
mode 'probe1'  : investigator's mj_probe2.run, noise 1.0    (saved: exp1 yaw +212.8, lo_turn [24,22])
mode 'indep'   : INDEPENDENT harness = turn_test.run_command loop + team_airlock.CpuPolicy (ONNX export) wrapped with
                 my own Gaussian noise std(model_5999.pt) * N(0,1) from a fresh RNG (seed 777); noise multiplier argv[2]
                 (0 = noise-free ONNX, reproduces the gate path); yaw reported from reset (turn_test convention) and
                 from turn start (probe convention); foot lift-offs counted from a policy-rate contact trace.
Reset seed 0 only (already used by v2 and the investigator; NOT a v2x seed)."""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/turning")
os.environ.setdefault("MJ_CACHE", str(HERE / "mjcache"))
import mj_probe2 as P

RUN, SEED, WZ = "arms-turn-turnboth-s1", 0, 0.6
mode = sys.argv[1]
t0 = time.time()
ck = P.ckpt_path(RUN, None)
if mode in ("probe0", "probe1"):  # indep_np = same harness, numpy actor from model_5999.pt
    o = P.run(ck, WZ, 3.0, 6.0, SEED, 0.0 if mode == "probe0" else 1.0)
    o.update(mode=mode, ckpt=ck, wall_s=round(time.time() - t0, 1))
else:
    import mujoco, torch
    from omegaconf import OmegaConf
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
    from bhl_robust.eval.multi_robot import build_multi
    from team_airlock import ContactRunner, CpuPolicy
    mult = float(sys.argv[2])
    run_dir = ck.rsplit("/", 1)[0]
    deploy = run_dir + "/exported/deploy.yaml"
    cfg = OmegaConf.load(deploy)
    std = torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]["std"].numpy().astype(np.float64)
    base = CpuPolicy(cfg.policy_checkpoint_path) if mode == "indep" else P.NpActor(ck, 0.0, 0, "all")
    rng = np.random.default_rng(777)

    class Noisy:
        def forward(self, obs):
            a = np.asarray(base.forward(obs), dtype=np.float64).reshape(1, -1)
            if mult > 0:
                a = a + mult * std[None, :] * rng.standard_normal(a.shape)
            return a.astype(np.float32)

    cache = Path(os.environ["MJ_CACHE"]) / "humanoid"
    model, slots = build_multi(Path(P.UPSTREAM), cache, 1, ["t"], variant="humanoid", world="flat")
    ctrl = RlController(cfg)
    ctrl.policy = Noisy()
    runner = ContactRunner(model, slots, [cfg], [ctrl])
    runner.reset(np.random.default_rng(SEED))
    slot = slots[0]
    owners = np.array([0 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "").startswith(slot.prefix)
                       else -1 for g in range(model.ngeom)])
    runner.configure_contacts(owners)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = {f: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{slot.prefix}leg_{f}_ankle_roll") for f in ("left", "right")}
    def contacts(d):
        on = {"left": False, "right": False}
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            o_ = g2 if g1 == floor else g1 if g2 == floor else None
            if o_ is None:
                continue
            for f in on:
                if int(model.geom_bodyid[o_]) == feet[f]:
                    on[f] = True
        return on
    dt = float(cfg.policy_dt)
    warm, secs = 3.0, 6.0
    yaw_reset = P.yaw_of(runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7])
    yaws, fell, trace = [], None, {"left": [], "right": []}
    yaw_turn0 = None
    for step in range(int((warm + secs) / dt)):
        now = step * dt
        c = np.zeros(3) if now < warm else np.array([0.0, 0.0, WZ])
        if now >= warm and yaw_turn0 is None:
            yaw_turn0 = len(yaws)          # index of the last pre-turn yaw sample (+1)
        obs = runner.observe(0, c)
        runner.step([ctrl.update(obs)])
        yaws.append(P.yaw_of(runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]))
        on = contacts(runner.d)
        if now >= warm:
            for f in on:
                trace[f].append(on[f])
        if runner.tilt(0) >= 0.78:
            fell = round(now, 2)
            break
    yv = np.unwrap(np.array([yaw_reset] + yaws))
    from_reset = math.degrees(yv[-1] - yv[0])
    from_turn = math.degrees(yv[-1] - yv[yaw_turn0])     # yv[yaw_turn0] = yaw after the last settle step
    o = {"mode": mode, "noise_mult": mult, "rng_seed": 777, "policy": cfg.policy_checkpoint_path, "std_ckpt": ck,
         "reset_seed": SEED, "wz": WZ, "fell_at_s": fell, "yaw_deg_from_reset": round(from_reset, 1),
         "yaw_deg_from_turn_start": round(from_turn, 1),
         "liftoffs_turn_policy_rate": [P.count_liftoffs(np.array(trace[f]), dt, min_air_s=dt) for f in ("left", "right")],
         "std_legs_mean": round(float(std[10:].mean()), 3), "wall_s": round(time.time() - t0, 1)}
print(json.dumps(o), flush=True)
with open(HERE / "rerun_results.jsonl", "a") as fh:
    fh.write(json.dumps(o) + "\n")
