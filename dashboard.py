#!/usr/bin/env python3
"""
Gesture Music Controller — Live Dashboard

Six-panel real-time dashboard showing gesture recognition state, model confidence,
IMU waveform, session history, and per-gesture health stats.

Usage:
    python dashboard.py
    python dashboard.py --port /dev/tty.usbserial-11130
"""

import argparse
import os
import re
import sys
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
import numpy as np
import serial
import serial.tools.list_ports

# ── Optional heavy imports (graceful fallback) ────────────────────────────────
try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

try:
    from pynput.keyboard import Key, Controller as KbController
    _keyboard = KbController()
    _HAS_PYNPUT = True
except Exception:
    _HAS_PYNPUT = False

# ── Configuration ─────────────────────────────────────────────────────────────
BAUD_RATE       = 115200
WINDOW_DURATION = 2.0   # seconds of IMU data per prediction window
COOLDOWN        = 1.5   # seconds between action triggers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "model", "gesture_model.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "model", "gesture_scaler.pkl")

LINE_RE = re.compile(
    r"AX:(?P<ax>[-\d.]+)\s+AY:(?P<ay>[-\d.]+)\s+AZ:(?P<az>[-\d.]+)"
    r"\s*\|\s*"
    r"GX:(?P<gx>[-\d.]+)\s+GY:(?P<gy>[-\d.]+)\s+GZ:(?P<gz>[-\d.]+)"
    r"\s*\|\s*"
    r"T:(?P<t>[-\d.]+)(?:\s*C)?"
)

# Gesture → (display name, action label, arrow char)
GESTURE_ACTIONS = {
    "up":           ("Raise up",   "Volume up",   "↑"),
    "down":         ("Push down",  "Volume down", "↓"),
    "rotate_right": ("Roll right", "Next track",  "↻"),
    "rotate_left":  ("Roll left",  "Prev track",  "↺"),
    "idle":         ("Idle",       "—",           "·"),
}

ALL_GESTURES     = ["idle", "up", "down", "rotate_left", "rotate_right"]
NON_IDLE         = ["up", "down", "rotate_right", "rotate_left"]

# ── Colour palette (matches plot_imu.py) ─────────────────────────────────────
C = {
    "bg":    "#0f1117",
    "panel": "#1a1d27",
    "grid":  "#2a2d3a",
    "ax":    "#4fc3f7",
    "ay":    "#81d4fa",
    "az":    "#b3e5fc",
    "gx":    "#f48fb1",
    "gy":    "#f06292",
    "gz":    "#e91e63",
    "text":  "#e0e0e0",
    "title": "#ffffff",
    "dim":   "#888888",
    "amber": "#f5a623",
    "green": "#69f0ae",
    "red":   "#ff5252",
}

GESTURE_COLORS = {
    "up":           "#f5a623",
    "down":         "#4fc3f7",
    "rotate_right": "#69f0ae",
    "rotate_left":  "#f06292",
    "idle":         "#888888",
}

# ── Utility functions ─────────────────────────────────────────────────────────

def find_port() -> str | None:
    ports = serial.tools.list_ports.comports()
    usb = [p for p in ports if "usb" in p.device.lower() or "serial" in p.device.lower()]
    if usb:
        return usb[0].device
    return ports[0].device if ports else None


def parse_line(line: str):
    m = LINE_RE.search(line)
    if m:
        return tuple(float(m.group(k)) for k in ("ax", "ay", "az", "gx", "gy", "gz", "t"))
    return None


def execute_command(gesture: str):
    if not _HAS_PYNPUT:
        return
    mapping = {
        "up":           Key.media_volume_up,
        "down":         Key.media_volume_down,
        "rotate_right": Key.media_next,
        "rotate_left":  Key.media_previous,
        "shake":        Key.media_play_pause,
        "push":         Key.media_volume_mute,
    }
    key = mapping.get(gesture)
    if key:
        _keyboard.tap(key)


# ── Shared state ──────────────────────────────────────────────────────────────

class SharedState:
    """All mutable dashboard state. Access only under self.lock."""

    def __init__(self):
        self.lock             = threading.Lock()
        self.last_gesture     = "idle"
        self.last_action      = "—"
        self.confidence_dict  = {g: 0.0 for g in ALL_GESTURES}
        self.session_log      = []               # list[str], capped at 6
        self.gesture_counts   = {g: 0   for g in NON_IDLE}
        self.gesture_conf_sums= {g: 0.0 for g in NON_IDLE}
        self.imu_buffer       = []               # list of [ax,ay,az,gx,gy,gz, ts]
        self.status           = "waiting"        # waiting|detecting|fired|cooldown
        self.serial_connected = False
        self.start_time       = time.perf_counter()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "last_gesture":      self.last_gesture,
                "last_action":       self.last_action,
                "confidence_dict":   dict(self.confidence_dict),
                "session_log":       list(self.session_log),
                "gesture_counts":    dict(self.gesture_counts),
                "gesture_conf_sums": dict(self.gesture_conf_sums),
                "imu_buffer":        list(self.imu_buffer),
                "status":            self.status,
                "serial_connected":  self.serial_connected,
                "start_time":        self.start_time,
            }


# ── Serial reader thread ──────────────────────────────────────────────────────

class LiveSerialReader(threading.Thread):
    """Background thread: reads ESP32 serial and fills shared.imu_buffer."""

    def __init__(self, port: str, baud: int, shared: SharedState):
        super().__init__(daemon=True)
        self.port   = port
        self.baud   = baud
        self.shared = shared

    def run(self):
        while True:
            try:
                with serial.Serial(self.port, self.baud, timeout=1) as ser:
                    with self.shared.lock:
                        self.shared.serial_connected = True
                    while True:
                        raw = ser.readline()
                        line = raw.decode("utf-8", errors="replace").strip()
                        parsed = parse_line(line)
                        if parsed:
                            row = list(parsed[:6]) + [time.perf_counter()]
                            with self.shared.lock:
                                self.shared.imu_buffer.append(row)
            except serial.SerialException:
                with self.shared.lock:
                    self.shared.serial_connected = False
                time.sleep(2)


# ── Prediction thread ─────────────────────────────────────────────────────────

class PredictionThread(threading.Thread):
    """Runs the 2-second windowed classifier loop; updates SharedState."""

    def __init__(self, model, scaler, shared: SharedState):
        super().__init__(daemon=True)
        self.model  = model
        self.scaler = scaler
        self.shared = shared

    def run(self):
        if self.model is None:
            return

        last_action_time = 0.0

        while True:
            with self.shared.lock:
                buf = list(self.shared.imu_buffer)

            now = time.perf_counter()

            # ── Cooldown ──
            if now - last_action_time < COOLDOWN:
                with self.shared.lock:
                    self.shared.imu_buffer.clear()
                    self.shared.status = "cooldown"
                time.sleep(0.05)
                continue

            if len(buf) < 2:
                with self.shared.lock:
                    self.shared.status = "detecting"
                time.sleep(0.01)
                continue

            buffer_duration = buf[-1][6] - buf[0][6]

            if buffer_duration < WINDOW_DURATION:
                with self.shared.lock:
                    self.shared.status = "detecting"
                time.sleep(0.01)
                continue

            # ── Feature extraction (mirrors live_control.py exactly) ──
            data_array = np.array(buf)[:, :6]
            features = []
            for col in range(6):
                axis_data = data_array[:, col]
                features.extend([
                    float(np.mean(axis_data)),
                    float(np.std(axis_data)),
                    float(np.max(axis_data)),
                    float(np.min(axis_data)),
                ])

            try:
                features_scaled = self.scaler.transform([features])
                prediction      = self.model.predict(features_scaled)[0]
                probas          = self.model.predict_proba(features_scaled)[0]
                conf_dict       = {cls: float(p) for cls, p in zip(self.model.classes_, probas)}
                confidence      = conf_dict.get(prediction, 0.0)
            except Exception as exc:
                print(f"[PredictionThread] inference error: {exc}", file=sys.stderr)
                time.sleep(0.01)
                continue

            if prediction != "idle":
                # Build log entry before acquiring lock
                elapsed = now - self.shared.start_time
                mm, ss  = divmod(int(elapsed), 60)
                disp, action, _ = GESTURE_ACTIONS.get(prediction, (prediction, prediction, ""))
                entry = f"{mm:02d}:{ss:02d}  {disp:<14}  {action:<14}  {confidence*100:.0f}%"

                with self.shared.lock:
                    self.shared.last_gesture      = prediction
                    self.shared.last_action       = action
                    self.shared.confidence_dict   = conf_dict
                    self.shared.status            = "fired"
                    self.shared.session_log       = (self.shared.session_log + [entry])[-6:]
                    if prediction in NON_IDLE:
                        self.shared.gesture_counts[prediction]    += 1
                        self.shared.gesture_conf_sums[prediction] += confidence
                    self.shared.imu_buffer.clear()

                # Execute media command outside the lock
                execute_command(prediction)
                last_action_time = time.perf_counter()

            else:
                # Slide window: remove oldest 20%
                slide = max(1, int(len(buf) * 0.2))
                with self.shared.lock:
                    self.shared.last_gesture    = "idle"
                    self.shared.confidence_dict = conf_dict
                    self.shared.status          = "detecting"
                    self.shared.imu_buffer      = self.shared.imu_buffer[slide:]

            time.sleep(0.01)


# ── Dashboard ─────────────────────────────────────────────────────────────────

class GestureDashboard:
    """Six-panel matplotlib dashboard driven by FuncAnimation."""

    def __init__(self, fig, shared: SharedState, no_model: bool, no_serial: bool):
        self.fig      = fig
        self._shared  = shared
        self._no_model  = no_model
        self._no_serial = no_serial

        self._card_patches    = {}
        self._card_name_texts = {}
        self._conf_bars       = None
        self._conf_labels     = []     # ordered gesture labels for bar chart
        self._conf_pct_texts  = []
        self._log_texts       = []
        self._waveform_lines  = {}
        self._health_count_texts = {}
        self._health_conf_texts  = {}

        self._build_layout()
        self._draw_instruction_cards()
        self._init_live_detection()
        self._init_confidence_bars()
        self._init_session_log()
        self._init_imu_waveform()
        self._init_health_stats()

        if no_model:
            self.fig.text(
                0.5, 0.5,
                "Model not found — run train_model.py first",
                ha="center", va="center", fontsize=16,
                color=C["red"], fontweight="bold",
                transform=self.fig.transFigure,
            )

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        gs = gridspec.GridSpec(
            5, 2,
            figure=self.fig,
            height_ratios=[2.2, 0.9, 2.2, 1.6, 1.4],
            hspace=0.55, wspace=0.32,
            left=0.05, right=0.97, top=0.93, bottom=0.05,
        )

        self.ax_instruction = self.fig.add_subplot(gs[0, :])
        self.ax_detection   = self.fig.add_subplot(gs[1, :])
        self.ax_confidence  = self.fig.add_subplot(gs[2, 0])
        self.ax_log         = self.fig.add_subplot(gs[2, 1])
        self.ax_waveform    = self.fig.add_subplot(gs[3, :])
        self.ax_health      = self.fig.add_subplot(gs[4, :])

        for ax in (self.ax_instruction, self.ax_detection, self.ax_confidence,
                   self.ax_log, self.ax_waveform, self.ax_health):
            ax.set_facecolor(C["panel"])
            for spine in ax.spines.values():
                spine.set_edgecolor(C["grid"])

        self.ax_instruction.set_title(
            "GESTURE INSTRUCTION SET", loc="left", fontsize=8,
            color=C["dim"], pad=6)
        self.ax_detection.set_title(
            "LIVE DETECTION", loc="left", fontsize=8,
            color=C["dim"], pad=4)
        self.ax_confidence.set_title(
            "CONFIDENCE BREAKDOWN", loc="left", fontsize=8,
            color=C["dim"], pad=4)
        self.ax_log.set_title(
            "SESSION LOG", loc="left", fontsize=8,
            color=C["dim"], pad=4)
        self.ax_waveform.set_title(
            "IMU LIVE WAVEFORM — ACCEL XYZ · GYRO XYZ", loc="left",
            fontsize=8, color=C["dim"], pad=4)
        self.ax_health.set_title(
            "GESTURE HEALTH — SESSION STATS", loc="left", fontsize=8,
            color=C["dim"], pad=4)

        for ax in (self.ax_instruction, self.ax_detection, self.ax_log, self.ax_health):
            ax.set_xticks([])
            ax.set_yticks([])

    # ── Panel 1: Instruction cards ───────────────────────────────────────────

    def _draw_instruction_cards(self):
        ax = self.ax_instruction
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        pad = 0.01
        n   = len(NON_IDLE)
        card_w = (1.0 / n) - 2 * pad

        for i, gesture in enumerate(NON_IDLE):
            x0 = i * (1.0 / n) + pad
            patch = FancyBboxPatch(
                (x0, 0.08), card_w, 0.80,
                boxstyle="round,pad=0.015",
                transform=ax.transAxes,
                facecolor=C["panel"],
                edgecolor=C["grid"],
                linewidth=1.2,
                clip_on=False,
                zorder=2,
            )
            ax.add_patch(patch)
            self._card_patches[gesture] = patch

            cx = x0 + card_w / 2
            disp, action, arrow = GESTURE_ACTIONS[gesture]
            color = GESTURE_COLORS[gesture]

            ax.text(cx, 0.63, arrow,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=24, color=color, zorder=3)

            name_txt = ax.text(cx, 0.39, disp,
                                transform=ax.transAxes, ha="center", va="center",
                                fontsize=9, color=C["text"], fontweight="bold", zorder=3)
            self._card_name_texts[gesture] = name_txt

            ax.text(cx, 0.18, action,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color=C["dim"], zorder=3)

    # ── Panel 2: Live detection ───────────────────────────────────────────────

    def _init_live_detection(self):
        ax = self.ax_detection
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        self._det_dot = ax.text(
            0.03, 0.5, "●",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=14, color=C["dim"])

        self._det_gesture = ax.text(
            0.5, 0.60, "—",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=26, fontweight="bold", color=C["dim"])

        self._det_action = ax.text(
            0.5, 0.22, "Waiting for gesture…",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=C["dim"])

        self._det_badge = ax.text(
            0.97, 0.5, "",
            transform=ax.transAxes, ha="right", va="center",
            fontsize=10, color=C["amber"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2a2d3a",
                      edgecolor=C["amber"], linewidth=1))

    # ── Panel 3: Confidence breakdown ────────────────────────────────────────

    def _init_confidence_bars(self):
        ax = self.ax_confidence
        # Order bars: non-idle first, then idle
        self._conf_labels = NON_IDLE + ["idle"]
        colors   = [GESTURE_COLORS[g] for g in self._conf_labels]
        y_pos    = list(range(len(self._conf_labels)))
        zeros    = [0.0] * len(self._conf_labels)
        disp_labels = [GESTURE_ACTIONS[g][0] for g in self._conf_labels]

        bars = ax.barh(disp_labels, zeros, color=colors, height=0.55,
                       align="center")
        self._conf_bars = bars

        ax.set_xlim(0, 1.15)
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(C["grid"])
        ax.xaxis.set_tick_params(labelcolor=C["dim"])
        ax.yaxis.set_tick_params(labelcolor=C["text"])
        ax.set_xlabel("Confidence", fontsize=8, color=C["dim"])
        ax.grid(True, axis="x", color=C["grid"], linewidth=0.5, linestyle="--")

        # Percentage labels to the right of each bar
        for i, _ in enumerate(self._conf_labels):
            txt = ax.text(0.02, i, "0%",
                          va="center", ha="left", fontsize=8,
                          color=C["text"])
            self._conf_pct_texts.append(txt)

    # ── Panel 4: Session log ──────────────────────────────────────────────────

    def _init_session_log(self):
        ax = self.ax_log
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        y_positions = [0.88, 0.72, 0.56, 0.40, 0.24, 0.08]
        for y in y_positions:
            txt = ax.text(
                0.04, y, "",
                transform=ax.transAxes, ha="left", va="center",
                fontsize=8, fontfamily="monospace", color=C["text"])
            self._log_texts.append(txt)

    # ── Panel 5: IMU waveform ─────────────────────────────────────────────────

    def _init_imu_waveform(self):
        ax = self.ax_waveform

        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text"], labelsize=8)
        ax.yaxis.label.set_color(C["text"])
        ax.xaxis.label.set_color(C["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["grid"])
        ax.grid(True, color=C["grid"], linewidth=0.5, linestyle="--")
        ax.set_xlabel("Time (s)", fontsize=8, color=C["dim"])
        ax.set_ylabel("Sensor value", fontsize=8, color=C["dim"])

        channels = [
            ("ax", C["ax"],  "Accel X"),
            ("ay", C["ay"],  "Accel Y"),
            ("az", C["az"],  "Accel Z"),
            ("gx", C["gx"],  "Gyro X"),
            ("gy", C["gy"],  "Gyro Y"),
            ("gz", C["gz"],  "Gyro Z"),
        ]
        for key, color, label in channels:
            line, = ax.plot([], [], color=color, lw=1.0, label=label, alpha=0.85)
            self._waveform_lines[key] = line

        ax.legend(
            loc="upper left", fontsize=7,
            facecolor=C["panel"], edgecolor=C["grid"],
            labelcolor=C["text"], ncol=2)

        ax.set_xlim(-WINDOW_DURATION - 0.5, 0.1)
        ax.set_ylim(-2, 2)

    # ── Panel 6: Gesture health stats ────────────────────────────────────────

    def _init_health_stats(self):
        ax = self.ax_health
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        pad    = 0.01
        n      = len(NON_IDLE)
        card_w = (1.0 / n) - 2 * pad

        for i, gesture in enumerate(NON_IDLE):
            x0    = i * (1.0 / n) + pad
            color = GESTURE_COLORS[gesture]

            patch = FancyBboxPatch(
                (x0, 0.08), card_w, 0.80,
                boxstyle="round,pad=0.015",
                transform=ax.transAxes,
                facecolor=C["panel"],
                edgecolor=C["grid"],
                linewidth=1.2,
                clip_on=False,
                zorder=2,
            )
            ax.add_patch(patch)

            cx = x0 + card_w / 2
            disp = GESTURE_ACTIONS[gesture][0]

            count_txt = ax.text(
                cx, 0.64, "0",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=22, fontweight="bold", color=color, zorder=3)
            self._health_count_texts[gesture] = count_txt

            ax.text(cx, 0.38, disp,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color=C["dim"], zorder=3)

            conf_txt = ax.text(
                cx, 0.18, "0% avg",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color=C["text"], zorder=3)
            self._health_conf_texts[gesture] = conf_txt

    # ── Animation update (main thread only) ──────────────────────────────────

    def update(self, frame: int):
        snap = self._shared.snapshot()
        self._update_instruction_highlight(snap)
        self._update_live_detection(snap)
        self._update_confidence_bars(snap)
        self._update_session_log(snap)
        self._update_imu_waveform(snap)
        self._update_health_stats(snap)

    def _update_instruction_highlight(self, snap: dict):
        active = snap["last_gesture"]
        for gesture, patch in self._card_patches.items():
            if gesture == active:
                patch.set_facecolor("#2a2410")
                patch.set_edgecolor(C["amber"])
                self._card_name_texts[gesture].set_color(C["amber"])
            else:
                patch.set_facecolor(C["panel"])
                patch.set_edgecolor(C["grid"])
                self._card_name_texts[gesture].set_color(C["text"])

    def _update_live_detection(self, snap: dict):
        gesture = snap["last_gesture"]
        status  = snap["status"]
        action  = snap["last_action"]
        connected = snap["serial_connected"]

        color = GESTURE_COLORS.get(gesture, C["dim"])
        disp  = GESTURE_ACTIONS.get(gesture, (gesture,))[0]

        self._det_gesture.set_text(disp.upper() if gesture != "idle" else "—")
        self._det_gesture.set_color(color if gesture != "idle" else C["dim"])

        if not connected and not self._no_serial:
            self._det_action.set_text("Waiting for device…")
            self._det_dot.set_color(C["dim"])
            self._det_badge.set_text("")
            return

        if status == "fired":
            dot_color = C["amber"]
        elif status == "detecting":
            dot_color = C["ax"]
        elif status == "cooldown":
            dot_color = C["red"]
        else:
            dot_color = C["dim"]

        self._det_dot.set_color(dot_color)
        self._det_action.set_text(action if gesture != "idle" else "Listening…")

        if gesture != "idle":
            conf = snap["confidence_dict"].get(gesture, 0.0)
            self._det_badge.set_text(f"{conf*100:.0f}%")
        else:
            self._det_badge.set_text("")

    def _update_confidence_bars(self, snap: dict):
        conf = snap["confidence_dict"]
        for i, (bar, gesture) in enumerate(zip(self._conf_bars, self._conf_labels)):
            val = conf.get(gesture, 0.0)
            bar.set_width(val)
            self._conf_pct_texts[i].set_x(val + 0.02)
            self._conf_pct_texts[i].set_text(f"{val*100:.0f}%")

    def _update_session_log(self, snap: dict):
        entries = list(reversed(snap["session_log"][-6:]))
        text_colors = [C["text"], "#bbbbbb", "#999999", "#777777", "#555555", "#444444"]
        for i, txt_artist in enumerate(self._log_texts):
            if i < len(entries):
                txt_artist.set_text(entries[i])
                txt_artist.set_color(text_colors[min(i, len(text_colors) - 1)])
            else:
                txt_artist.set_text("")

    def _update_imu_waveform(self, snap: dict):
        buf = snap["imu_buffer"]
        if len(buf) < 2:
            return

        t_now = buf[-1][6]
        t_lo  = t_now - WINDOW_DURATION - 1.0   # show 1 extra second of history

        rows = [r for r in buf if r[6] >= t_lo]
        if len(rows) < 2:
            return

        t_rel = np.array([r[6] - t_now for r in rows])  # 0 = now, negative = past
        channels_idx = {"ax": 0, "ay": 1, "az": 2, "gx": 3, "gy": 4, "gz": 5}

        for key, line in self._waveform_lines.items():
            vals = np.array([r[channels_idx[key]] for r in rows])
            line.set_data(t_rel, vals)

        x_lo = -(WINDOW_DURATION + 1.0)
        self.ax_waveform.set_xlim(x_lo, 0.1)

        all_vals = np.concatenate([
            np.array([r[j] for r in rows]) for j in range(6)
        ])
        if len(all_vals):
            lo, hi = all_vals.min(), all_vals.max()
            pad = max((hi - lo) * 0.15, 0.5)
            self.ax_waveform.set_ylim(lo - pad, hi + pad)

    def _update_health_stats(self, snap: dict):
        counts    = snap["gesture_counts"]
        conf_sums = snap["gesture_conf_sums"]
        for gesture in NON_IDLE:
            n   = counts.get(gesture, 0)
            avg = conf_sums.get(gesture, 0.0) / max(n, 1)
            self._health_count_texts[gesture].set_text(str(n))
            self._health_conf_texts[gesture].set_text(
                f"{avg*100:.0f}% avg" if n > 0 else "—")


# ── Demo mode ─────────────────────────────────────────────────────────────────

class DemoThread(threading.Thread):
    """Cycles through fake gesture events so the dashboard looks alive."""

    DEMO_SEQUENCE = [
        ("up",           "Volume up",   {"idle": 0.04, "up": 0.52, "down": 0.09, "rotate_left": 0.18, "rotate_right": 0.17}),
        ("rotate_right", "Next track",  {"idle": 0.05, "up": 0.11, "down": 0.08, "rotate_left": 0.10, "rotate_right": 0.66}),
        ("down",         "Volume down", {"idle": 0.06, "up": 0.07, "down": 0.55, "rotate_left": 0.16, "rotate_right": 0.16}),
        ("rotate_left",  "Prev track",  {"idle": 0.03, "up": 0.09, "down": 0.12, "rotate_left": 0.73, "rotate_right": 0.03}),
        ("up",           "Volume up",   {"idle": 0.02, "up": 0.60, "down": 0.10, "rotate_left": 0.14, "rotate_right": 0.14}),
        ("rotate_right", "Next track",  {"idle": 0.04, "up": 0.08, "down": 0.07, "rotate_left": 0.07, "rotate_right": 0.74}),
    ]

    def __init__(self, shared: SharedState):
        super().__init__(daemon=True)
        self.shared = shared

    def run(self):
        # Seed IMU buffer with enough history for the waveform
        t0 = time.perf_counter()
        for i in range(400):
            t = t0 - 4.0 + i * 0.01
            row = [
                0.15 * np.sin(i * 0.15) + 0.04 * np.random.randn(),
                0.10 * np.cos(i * 0.12) + 0.04 * np.random.randn(),
                9.78 + 0.06 * np.sin(i * 0.08),
                12.0 * np.sin(i * 0.10) + 0.8 * np.random.randn(),
                -8.0 * np.cos(i * 0.13) + 0.8 * np.random.randn(),
                3.5  * np.sin(i * 0.09),
                t,
            ]
            with self.shared.lock:
                self.shared.imu_buffer.append(row)

        with self.shared.lock:
            self.shared.serial_connected = True

        step = 0
        while True:
            # "Detecting" phase — waveform scrolls, status dot blue
            with self.shared.lock:
                self.shared.status = "detecting"
                self.shared.last_gesture = "idle"
                self.shared.last_action  = "Listening…"
                self.shared.confidence_dict = {g: 0.0 for g in ALL_GESTURES}

            # Keep advancing the waveform for ~2.5 seconds
            for _ in range(50):
                t_now = time.perf_counter()
                row = [
                    0.15 * np.sin(t_now * 3.0),
                    0.10 * np.cos(t_now * 2.5),
                    9.78 + 0.06 * np.sin(t_now * 1.5),
                    12.0 * np.sin(t_now * 2.0),
                    -8.0 * np.cos(t_now * 2.3),
                    3.5  * np.sin(t_now * 1.8),
                    t_now,
                ]
                with self.shared.lock:
                    self.shared.imu_buffer.append(row)
                    # keep buffer bounded
                    if len(self.shared.imu_buffer) > 600:
                        self.shared.imu_buffer = self.shared.imu_buffer[-400:]
                time.sleep(0.05)

            # "Fired" phase — show gesture result
            gesture, action, conf_dict = self.DEMO_SEQUENCE[step % len(self.DEMO_SEQUENCE)]
            confidence = conf_dict[gesture]

            elapsed = time.perf_counter() - self.shared.start_time
            mm, ss  = divmod(int(elapsed), 60)
            disp    = GESTURE_ACTIONS[gesture][0]
            entry   = f"{mm:02d}:{ss:02d}  {disp:<14}  {action:<14}  {confidence*100:.0f}%"

            with self.shared.lock:
                self.shared.last_gesture     = gesture
                self.shared.last_action      = action
                self.shared.confidence_dict  = conf_dict
                self.shared.status           = "fired"
                self.shared.session_log      = (self.shared.session_log + [entry])[-6:]
                if gesture in NON_IDLE:
                    self.shared.gesture_counts[gesture]    += 1
                    self.shared.gesture_conf_sums[gesture] += confidence

            time.sleep(1.5)

            # "Cooldown" phase
            with self.shared.lock:
                self.shared.status = "cooldown"
            time.sleep(1.0)

            step += 1


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gesture Music Controller Dashboard")
    parser.add_argument("--port", default=None, help="Serial port (auto-detected if omitted)")
    parser.add_argument("--demo", action="store_true", help="Run with simulated data (no ESP32 needed)")
    args = parser.parse_args()

    shared = SharedState()

    if args.demo:
        print("[INFO] Running in demo mode — no ESP32 or model required")
        DemoThread(shared).start()
        no_model, no_serial = False, False
    else:
        # Load model
        no_model = False
        model, scaler = None, None
        if _HAS_JOBLIB:
            try:
                model  = joblib.load(MODEL_PATH)
                scaler = joblib.load(SCALER_PATH)
                print(f"[INFO] Model loaded from {MODEL_PATH}")
            except Exception as e:
                print(f"[WARN] Could not load model: {e}", file=sys.stderr)
                no_model = True
        else:
            print("[WARN] joblib not installed — model inference disabled", file=sys.stderr)
            no_model = True

        # Find serial port
        port = args.port or find_port()
        no_serial = port is None
        if no_serial:
            print("[WARN] No serial port found — running in display-only mode", file=sys.stderr)
        else:
            print(f"[INFO] Using serial port: {port}")

        if not no_serial:
            LiveSerialReader(port, BAUD_RATE, shared).start()

        PredictionThread(model, scaler, shared).start()

    # ── Build figure ──────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), facecolor=C["bg"])
    try:
        fig.canvas.manager.set_window_title("Gesture Music Controller")
    except Exception:
        pass

    fig.suptitle(
        "Gesture Music Controller",
        color=C["title"], fontsize=13, fontweight="bold", y=0.975)

    dashboard = GestureDashboard(fig, shared, no_model=no_model, no_serial=no_serial)

    ani = FuncAnimation(  # noqa: F841  keep reference alive
        fig, dashboard.update,
        interval=50, blit=False, cache_frame_data=False)

    plt.show()


if __name__ == "__main__":
    main()
