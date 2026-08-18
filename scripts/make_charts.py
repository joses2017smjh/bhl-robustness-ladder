"""Render the README's static SVG charts, light and dark variants.

Two files per chart so the README can use <picture> with prefers-color-scheme;
SVG served as <img> on GitHub does not honour a media query inside the file.

Colour roles follow the validated palette:
  - DR rungs are an ORDINAL ramp (randomization strength is a magnitude, not an
    identity), single blue hue, monotone lightness. Validated with --ordinal in
    both modes.
  - Push arms are two CATEGORICAL slots (blue/orange), validated --pairs all.
Text never wears a series colour.
"""
from __future__ import annotations
import json, math
from pathlib import Path

THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8880",
                  grid="#e6e5e1", axis="#c9c7c1",
                  ramp=["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
                  cat=["#2a78d6", "#eb6834"], bad="#e34948"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#8f8e85",
                 grid="#2e2e2b", axis="#45443f",
                 ramp=["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
                 cat=["#3987e5", "#d95926"], bad="#e66767"),
}
FONT = "ui-sans-serif,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"


def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Chart:
    def __init__(self, w, h, m, t):
        self.w, self.h, self.m, self.t = w, h, m, t
        self.o = []
    @property
    def pw(self): return self.w - self.m["l"] - self.m["r"]
    @property
    def ph(self): return self.h - self.m["t"] - self.m["b"]
    def sx(self, v): return self.m["l"] + (v - self.x0) / (self.x1 - self.x0) * self.pw
    def sy(self, v): return self.m["t"] + (1 - (v - self.y0) / (self.y1 - self.y0)) * self.ph
    def add(self, s): self.o.append(s)

    def frame(self, xticks, yticks, xlabel, ylabel, xfmt=str, yfmt=str):
        t = self.t
        for v in yticks:
            y = self.sy(v)
            self.add(f'<line x1="{self.m["l"]}" y1="{y:.1f}" x2="{self.m["l"]+self.pw}" y2="{y:.1f}" '
                     f'stroke="{t["grid"]}" stroke-width="1"/>')
            self.add(f'<text x="{self.m["l"]-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" '
                     f'fill="{t["ink3"]}" font-family="{FONT}">{esc(yfmt(v))}</text>')
        for v in xticks:
            x = self.sx(v)
            self.add(f'<text x="{x:.1f}" y="{self.m["t"]+self.ph+20}" text-anchor="middle" font-size="12" '
                     f'fill="{t["ink3"]}" font-family="{FONT}">{esc(xfmt(v))}</text>')
        self.add(f'<line x1="{self.m["l"]}" y1="{self.m["t"]+self.ph}" x2="{self.m["l"]+self.pw}" '
                 f'y2="{self.m["t"]+self.ph}" stroke="{t["axis"]}" stroke-width="1"/>')
        self.add(f'<text x="{self.m["l"]+self.pw/2:.0f}" y="{self.m["t"]+self.ph+44:.0f}" text-anchor="middle" '
                 f'font-size="12.5" fill="{t["ink2"]}" font-family="{FONT}">{esc(xlabel)}</text>')
        self.add(f'<text transform="translate({self.m["l"]-48:.0f},{self.m["t"]+self.ph/2:.0f}) rotate(-90)" '
                 f'text-anchor="middle" font-size="12.5" fill="{t["ink2"]}" '
                 f'font-family="{FONT}">{esc(ylabel)}</text>')

    def line(self, pts, colour, width=2, dash=None, opacity=1.0):
        if not pts: return
        d = "M" + " L".join(f"{self.sx(x):.1f},{self.sy(y):.1f}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}" '
                 f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"{da}/>')

    def title(self, text, sub=None):
        self.add(f'<text x="{self.m["l"]-40}" y="26" font-size="15.5" font-weight="600" '
                 f'fill="{self.t["ink"]}" font-family="{FONT}">{esc(text)}</text>')
        if sub:
            self.add(f'<text x="{self.m["l"]-40}" y="44" font-size="12.5" fill="{self.t["ink2"]}" '
                     f'font-family="{FONT}">{esc(sub)}</text>')

    def render(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img">'
                f'<rect width="{self.w}" height="{self.h}" fill="{self.t["surface"]}"/>'
                + "".join(self.o) + "</svg>")


D = json.loads(Path("results/curves/curves.json").read_text())
curves, summary = D["curves"], {s["label"]: s for s in D["summary"]}
OUT = Path("results/charts"); OUT.mkdir(parents=True, exist_ok=True)

DR_RUNGS = [("0.0", "dr-off"), ("0.5", "dr-s0.5"), ("1.0", "dr-default"),
            ("1.5", "dr-s1.5"), ("2.0", "dr-aggressive")]


def mean_curve(prefix, metric):
    """Seed-averaged curve on a common step grid."""
    series = [curves[l][metric] for l in curves if l.rsplit("-s", 1)[0] == prefix and metric in curves[l]]
    if not series: return []
    n = min(len(s) for s in series)
    return [[series[0][i][0], sum(s[i][1] for s in series) / len(series)] for i in range(n)]


def chart_dr_curves(mode):
    t = THEME[mode]
    c = Chart(880, 430, dict(l=68, r=132, t=62, b=52), t)
    c.x0, c.x1, c.y0, c.y1 = 0, 6000, 0, 55
    c.title("Domain randomization: training reward by fidelity rung",
            "Seed-averaged. Higher rung = wider randomization. s=0.5 and s=1.5 still training.")
    c.frame([0, 1500, 3000, 4500, 6000], [0, 10, 20, 30, 40, 50],
            "PPO iteration", "mean reward", xfmt=lambda v: f"{v:,}")
    for i, (scale, prefix) in enumerate(DR_RUNGS):
        pts = mean_curve(prefix, "reward")
        if not pts: continue
        partial = pts[-1][0] < 5800
        c.line(pts, t["ramp"][i], dash="5 4" if partial else None)
        lx, ly = c.sx(pts[-1][0]) + 8, c.sy(pts[-1][1]) + 4
        lbl = f"s={scale}" + ("  (running)" if partial else "")
        c.add(f'<text x="{lx:.0f}" y="{ly:.0f}" font-size="12" fill="{t["ink2"]}" '
              f'font-family="{FONT}">{esc(lbl)}</text>')
    c.add(f'<text x="{c.m["l"]+8}" y="{c.m["t"]+16}" font-size="11.5" fill="{t["ink3"]}" '
          f'font-family="{FONT}">dashed = run still in progress</text>')
    return c.render()


def chart_push(mode):
    t = THEME[mode]
    c = Chart(880, 430, dict(l=68, r=150, t=62, b=52), t)
    c.x0, c.x1, c.y0, c.y1 = 0, 6000, 0, 28
    c.title("Push recovery, pass 1: the curriculum learned, then the ramp destroyed it",
            "Ceiling was 1.5 m/s. Collapse begins as the ramp passes ~0.7 m/s.")
    c.frame([0, 1500, 3000, 4500, 6000], [0, 5, 10, 15, 20, 25],
            "PPO iteration", "mean reward", xfmt=lambda v: f"{v:,}")
    # Both arms converge to ~3, so their end labels would overlap. Collect the
    # anchors, then push them apart before drawing.
    ends = []
    for i, (prefix, name) in enumerate([("push-curriculum", "curriculum"), ("push-fixed", "fixed 1.5 m/s")]):
        pts = mean_curve(prefix, "reward")
        c.line(pts, t["cat"][i])
        if pts:
            ends.append([c.sy(pts[-1][1]) + 4, c.sx(pts[-1][0]) + 10, name, t["cat"][i]])
    ends.sort()
    for j in range(1, len(ends)):
        if ends[j][0] - ends[j - 1][0] < 16:
            ends[j][0] = ends[j - 1][0] + 16
    for y, x, name, colour in ends:
        # A short leader dot in the series colour carries identity; the text
        # itself stays in ink, never the series colour.
        c.add(f'<circle cx="{x-4:.0f}" cy="{y-4:.0f}" r="3.5" fill="{colour}"/>')
        c.add(f'<text x="{x+4:.0f}" y="{y:.0f}" font-size="12" fill="{t["ink2"]}" '
              f'font-family="{FONT}">{esc(name)}</text>')
    peak = max(mean_curve("push-curriculum", "reward") or [[0, 0]], key=lambda p: p[1])
    px, py = c.sx(peak[0]), c.sy(peak[1])
    c.add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="none" stroke="{t["bad"]}" stroke-width="2"/>')
    c.add(f'<text x="{px+10:.0f}" y="{py-8:.0f}" font-size="11.5" fill="{t["ink2"]}" '
          f'font-family="{FONT}">peak {peak[1]:.1f} @ iter {peak[0]:,} — then collapses</text>')
    return c.render()


def chart_ladder(mode):
    """Two panels sharing the x variable. Never two y-scales on one plot."""
    t = THEME[mode]
    W, H = 880, 400
    # Explicit panel boxes; each panel owns the gutter to its left so the
    # rotated y-label cannot land on the neighbouring panel.
    PANELS = [
        dict(left=76,  width=316, ylabel="final mean reward", metric="reward", ymax=55.0,
             yt=[0, 10, 20, 30, 40, 50], yfmt=lambda v: f"{v:g}"),
        dict(left=536, width=316, ylabel="fall rate (fraction of episodes)", metric="fall_frac", ymax=1.0,
             yt=[0, 0.25, 0.5, 0.75, 1.0], yfmt=lambda v: f"{v:.2f}"),
    ]
    o = [f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>']
    o.append(f'<text x="28" y="28" font-size="15.5" font-weight="600" fill="{t["ink"]}" '
             f'font-family="{FONT}">The fidelity ladder: performance and stability vs randomization strength</text>')
    o.append(f'<text x="28" y="47" font-size="12.5" fill="{t["ink2"]}" font-family="{FONT}">'
             f'Each point is one seed; the line joins rung means. Hollow markers are runs still training.</text>')

    for P in PANELS:
        c = Chart(W, H, dict(l=P["left"], r=W - P["left"] - P["width"], t=78, b=76), t)
        c.x0, c.x1, c.y0, c.y1 = -0.18, 2.18, 0, P["ymax"]
        c.frame([0, 0.5, 1.0, 1.5, 2.0], P["yt"], "DR scale  s", P["ylabel"],
                xfmt=lambda v: f"{v:g}", yfmt=P["yfmt"])
        means = []
        for i, (scale, prefix) in enumerate(DR_RUNGS):
            vals, done = [], True
            for lbl, sm in summary.items():
                if lbl.rsplit("-s", 1)[0] == prefix:
                    v = sm["final_reward"] if P["metric"] == "reward" else sm["final_fall"]
                    if v is not None:
                        vals.append(v); done &= sm["complete"]
            if not vals: continue
            sx = float(scale)
            means.append((sx, sum(vals) / len(vals)))
            for v in vals:
                c.add(f'<circle cx="{c.sx(sx):.1f}" cy="{c.sy(v):.1f}" r="4.5" '
                      f'fill="{t["ramp"][i] if done else t["surface"]}" '
                      f'stroke="{t["ramp"][i]}" stroke-width="2"/>')
        c.line(sorted(means), t["ramp"][2], width=2, opacity=0.5)
        o.extend(c.o)

    o.append(f'<text x="28" y="{H-12}" font-size="11.5" fill="{t["ink3"]}" font-family="{FONT}">'
             f'The cliff sits between s=1.0 and s=1.5: reward falls ~4x and the fall rate goes from 0.04 to 0.51.</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'role="img">' + "".join(o) + "</svg>")


for name, fn in [("dr_training_curves", chart_dr_curves), ("push_collapse", chart_push),
                 ("dr_ladder_summary", chart_ladder)]:
    for mode in ("light", "dark"):
        p = OUT / f"{name}-{mode}.svg"
        p.write_text(fn(mode))
        print(f"  {p}  ({p.stat().st_size//1024} KB)")
