"""Per-maze outcome matrix of the sysid gym re-evaluation (data/gymcheck/*.json; training-range maze seeds 9000-9047 only),
with paired (same maze, same start position) discordance against the default-gym 'base' run and an exact two-sided
McNemar p-value. Writes data/gymcheck_matrix.json."""
import glob, json, math
from pathlib import Path

D = Path(__file__).resolve().parent / "data"
runs = {}
for f in sorted(glob.glob(str(D / "gymcheck" / "*.json"))):
    d = json.loads(Path(f).read_text())
    s = d["summary"]
    runs.setdefault((s["arm"], s["maze_seeds"][0]), {})[s["config"]] = {e["maze_seed"]: e["outcome"] for e in d["episodes"]}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


out = {}
for (arm, start), cfgs in sorted(runs.items()):
    seeds = sorted(next(iter(cfgs.values())))
    print(f"== {arm} maze seeds {seeds[0]}-{seeds[-1]}")
    print(" " * 44 + " ".join(str(s)[-2:] for s in seeds) + "   G  C  T | vs base: lost won p")
    base = cfgs.get("base")
    block = {}
    for c in sorted(cfgs, key=lambda c: (c != "base", len(c), c)):
        o = cfgs[c]
        g = sum(o[s] == "goal" for s in seeds); col = sum(o[s] == "collision" for s in seeds); to = sum(o[s] == "time_out" for s in seeds)
        row = {"goal": g, "collision": col, "time_out": to, "per_maze": {str(s): o[s] for s in seeds}}
        tail = ""
        if base is not None and c != "base":
            lost = sum(base[s] == "goal" and o[s] != "goal" for s in seeds)
            won = sum(base[s] != "goal" and o[s] == "goal" for s in seeds)
            row.update(lost_vs_base=lost, won_vs_base=won, mcnemar_p=round(mcnemar(lost, won), 3))
            tail = f" | {lost:4d} {won:4d} {mcnemar(lost, won):.3f}"
        print(f"{c:44s}" + " ".join(" " + o[s][0].upper() for s in seeds) + f"  {g:2d} {col:2d} {to:2d}" + tail)
        block[c] = row
    out[f"{arm}__{seeds[0]}-{seeds[-1]}"] = block
(D / "gymcheck_matrix.json").write_text(json.dumps(out, indent=1))
