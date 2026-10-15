"""Display only (not a gate): a WaiterEnv that films each episode for a clip.

Used through trace_base.py on dev seeds:
  trace_base.py --env import --env-target clip_env:RecordingWaiterEnv --env-kwargs '{<WaiterEnv kwargs>,
      "record_dir": "<dir>", "label": "<what is driving>"}' ...
Every env step (0.08 s) it renders a third-person view that follows the robot and the head camera the policy sees,
and writes them with the instruction, the time and the stages reached to <record_dir>/<label>_seed<seed>.mp4, plus a
.json with the episode's outcome. The renderers only read the simulation state, so the episode is the one the policy
runs without the recording.

swap_target=True (2026-10-15, the two-instruction clip): after the reset, the episode's target colour and target table
are both swapped to the other ones on the SAME physical layout, and the instruction is rewritten with the same
template. Every colour/table combination occurs in the layout distribution, and success is scored against the swapped
target (WaiterEnv.evaluate reads self.layout). The tables carry no marking of the target, so only the instruction
says which tumbler and which table.
"""
from __future__ import annotations

import dataclasses
import json
import os
import socket
import textwrap
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from bhl_robust.eval.panels import BG, DIM, PANEL_BG, TEXT, load_font
from bhl_robust.eval import waiter_scene as W
from bhl_robust.eval.waiter_env import IMAGE_HW, STAGES, STEP_DT, WaiterEnv, instruction_text

STAGE_TEXT = ("tumbler lifted", "tumbler on the tray", "tray lifted", "tray at the table", "tray on the target",
              "held 1 s: success")
GOOD, BAD = (30, 110, 52), (150, 34, 30)
SIDE_W, BAR_H = 250, 32


class RecordingWaiterEnv(WaiterEnv):
    def __init__(self, record_dir, label="policy", view=(560, 420), cam=(2.4, -30.0, 200.0), swap_target=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.swap_target = bool(swap_target)
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.label = str(label)
        self.view = (int(view[0]), int(view[1]))
        self.cam_cfg = tuple(float(v) for v in cam)        # distance, elevation, world azimuth (deg)
        self._view_r = self._head_r = self._writer = None
        self._rec = None

    # ------------------------------------------------------------------------------------------------ contract API
    def reset(self, seed, instruction_id=None):
        self._finish(None)
        obs = super().reset(seed, instruction_id=instruction_id)
        original = {"target_colour": self.layout.target_colour, "target_side": self.layout.target_side}
        if self.swap_target:
            other = lambda v, pool: [x for x in pool if x != v][0]                      # noqa: E731
            self.layout = dataclasses.replace(self.layout,
                                              target_colour=other(self.layout.target_colour, W.COLOURS),
                                              target_side=other(self.layout.target_side, W.SIDES))
            self.instruction, self.template_id = instruction_text(self.layout, self.template_id)
            obs = self._obs()
        import imageio.v2 as imageio
        path = self.record_dir / f"{self.label}_seed{int(seed)}.mp4"
        self._writer = imageio.get_writer(str(path), fps=1.0 / STEP_DT, codec="libx264", quality=8,
                                          pixelformat="yuv420p", macro_block_size=1)
        self._rec = {"label": self.label, "seed": int(seed), "instruction": self.instruction,
                     "template": self.template_id, "layout": self.layout.as_dict(), "mp4": str(path),
                     "deploy": str(self.deploy), "host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                     "frames": 0, "display_only": True, "swap_target": self.swap_target,
                     "layout_target": original}
        a = self._sl.qpos_adr
        self._look = np.array([0.5 * float(self.d.qpos[a]) + 0.2, float(self.d.qpos[a + 1])])
        self._frame({"stage": 0, "t": 0.0})
        return obs

    def step(self, a):
        obs, r, done, info = super().step(a)
        self._frame(info)
        if done:
            self._finish(info)
        return obs, r, done, info

    def close(self):
        self._finish(None)
        for r in (self._view_r, self._head_r):
            if r is not None:
                r.close()
        self._view_r = self._head_r = None
        super().close()

    # ------------------------------------------------------------------------------------------------ recording
    def _render(self):
        if self._view_r is None:
            self._view_r = mujoco.Renderer(self.model, height=self.view[1], width=self.view[0])
            self._head_r = mujoco.Renderer(self.model, height=IMAGE_HW, width=IMAGE_HW)
        a = self._sl.qpos_adr
        base = np.array([float(self.d.qpos[a]), float(self.d.qpos[a + 1])])
        self._look += 0.12 * (base - self._look)          # follow the robot without the step sway
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.elevation, cam.azimuth = self.cam_cfg
        cam.lookat[:] = [self._look[0], self._look[1], 0.45]
        self._view_r.update_scene(self.d, camera=cam)
        view = self._view_r.render().copy()
        self._head_r.update_scene(self.d, camera=self.ids.cam)
        return view, self._head_r.render().copy()

    def _compose(self, view, head, info, card=None) -> np.ndarray:
        w, h = self.view
        W, H = w + SIDE_W, h + 2 * BAR_H
        img = Image.new("RGB", (W + W % 2, H + H % 2), BG)
        img.paste(Image.fromarray(view), (0, BAR_H))
        draw = ImageDraw.Draw(img)
        draw.text((8, 7), f"{self.label} | dev layout {self._rec['seed']} | t = {float(info.get('t', 0.0)):4.1f} s",
                  font=load_font(15), fill=TEXT)
        draw.text((8, H - BAR_H + 9), "simulation | stance lock while standing | display only, not a gate",
                  font=load_font(12), fill=DIM)
        x0, y = w + 8, BAR_H
        draw.rectangle((w, BAR_H, W, H - BAR_H), fill=PANEL_BG)
        draw.text((x0, y + 4), "head camera (policy input)", font=load_font(12), fill=DIM)
        hs = SIDE_W - 16
        img.paste(Image.fromarray(head).resize((hs, hs), Image.BILINEAR), (x0, y + 22))
        y += 22 + hs + 8
        for line in textwrap.wrap(f"“{self.instruction}”", 30)[:4]:
            draw.text((x0, y), line, font=load_font(12), fill=TEXT)
            y += 16
        y += 6
        stage = int(info.get("stage", 0))
        for i, s in enumerate(STAGE_TEXT, start=1):
            done = stage >= i
            draw.text((x0, y), ("✓ " if done else "· ") + s, font=load_font(12),
                      fill=(110, 210, 120) if done else DIM)
            y += 16
        if card:
            text, ok = card
            font = load_font(18)
            tw = draw.textlength(text, font=font)
            cx, cy = (w - tw) / 2 - 12, BAR_H + h - 54
            draw.rectangle((cx, cy, cx + tw + 24, cy + 38), fill=GOOD if ok else BAD, outline=(255, 255, 255), width=2)
            draw.text((cx + 12, cy + 9), text, font=font, fill=(255, 255, 255))
        return np.asarray(img)

    def _frame(self, info):
        if self._writer is None:
            return
        view, head = self._render()
        self._last = (view, head)
        self._writer.append_data(self._compose(view, head, info))
        self._rec["frames"] += 1

    def _finish(self, info):
        if self._writer is None:
            return
        if info is not None:
            ok, fell, stage = bool(info.get("success")), bool(info.get("fell")), int(info.get("stage", 0))
            text = (f"SUCCESS at {info['t']:.1f} s" if ok else f"FELL at {info['t']:.1f} s" if fell else
                    f"time limit: furthest stage '{STAGES[stage].replace('_', ' ')}'")
            frame = self._compose(*self._last, info, card=(text, ok))
            for _ in range(int(round(2.0 / STEP_DT))):
                self._writer.append_data(frame)
            self._rec.update({"success": ok, "fell": fell, "stage": stage, "stage_name": STAGES[stage],
                              "t_end": float(info.get("t", 0.0)), "result_card": text})
        else:
            self._rec["interrupted"] = True
        self._writer.close()
        self._writer = None
        Path(self._rec["mp4"]).with_suffix(".json").write_text(json.dumps(self._rec, indent=1) + "\n")
