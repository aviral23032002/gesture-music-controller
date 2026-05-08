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
import math
import subprocess
from live_detect import Madgwick

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
WINDOW_DURATION = 2.5   # seconds of IMU data per prediction window
COOLDOWN        = 1.5   # seconds between action triggers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "model", "gesture_model_2.5sec.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "model", "gesture_scaler_2.5sec.pkl")

LINE_RE = re.compile(
    r"AX:(?P<ax>[-\d.]+)\s+AY:(?P<ay>[-\d.]+)\s+AZ:(?P<az>[-\d.]+)"
    r"\s*\|\s*"
    r"GX:(?P<gx>[-\d.]+)\s+GY:(?P<gy>[-\d.]+)\s+GZ:(?P<gz>[-\d.]+)"
    r"\s*\|\s*"
    r"T:(?P<t>[-\d.]+)(?:\s*C)?"
)

# Gesture → (display name, action label, arrow char)
GESTURE_ACTIONS = {
    "up":                 ("Raise up",    "Volume up",   "↑"),
    "down":               ("Push down",   "Volume down", "↓"),
    "left":               ("Swipe left",  "Prev track",  "←"),
    "right":              ("Swipe right", "Next track",  "→"),
    "push":               ("Push fwd",    "Play/Pause",  "→|"),
    "pull":               ("Pull back",   "Play/Pause",  "|←"),
    "wrist_rotate_right": ("Roll right",  "+10s",        "↻"),
    "wrist_rotate_left":  ("Roll left",   "-10s",        "↺"),
    "clap":               ("Clap",        "Wake",        "><"),
    "idle":               ("Idle",        "—",           "·"),
}

ALL_GESTURES     = ["idle", "up", "down", "left", "right", "push", "pull", "wrist_rotate_left", "wrist_rotate_right", "clap"]
NON_IDLE         = ["up", "down", "left", "right", "push", "pull", "wrist_rotate_left", "wrist_rotate_right", "clap"]

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
    "up":                 "#f5a623",
    "down":               "#4fc3f7",
    "left":               "#ffcc80",
    "right":              "#ce93d8",
    "push":               "#ff8a65",
    "pull":               "#aed581",
    "wrist_rotate_right": "#69f0ae",
    "wrist_rotate_left":  "#f06292",
    "clap":               "#ffd54f",
    "idle":               "#888888",
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


def send_spotify_command(cmd: str):
    try:
        subprocess.run(["osascript", "-e", f'tell application "Spotify" to {cmd}'], check=False)
    except Exception:
        pass

def get_spotify_info():
    try:
        script = '''
        if application "Spotify" is running then
            tell application "Spotify"
                if player state is stopped then return "Stopped|"
                set currentArtist to artist of current track
                set currentTrack to name of current track
                set playerState to player state as string
                return currentTrack & " - " & currentArtist & "|" & playerState
            end tell
        else
            return "Not running|Stopped"
        end if
        '''
        res = subprocess.check_output(["osascript", "-e", script], timeout=1)
        res = res.decode("utf-8").strip()
        parts = res.split("|")
        if len(parts) == 2:
            return parts[0], parts[1]
        return res, ""
    except Exception:
        return "Unknown", "Stopped"

def execute_command(gesture: str):
    if gesture in ["push", "pull"]:
        send_spotify_command("playpause")
    elif gesture == "right":
        send_spotify_command("next track")
    elif gesture == "left":
        send_spotify_command("previous track")
    elif gesture == "up":
        send_spotify_command("set sound volume to (sound volume + 10)")
    elif gesture == "down":
        send_spotify_command("set sound volume to (sound volume - 10)")
    elif gesture == "wrist_rotate_right":
        send_spotify_command("set player position to (player position + 10)")
    elif gesture == "wrist_rotate_left":
        send_spotify_command("set player position to (player position - 10)")


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
        self.spotify_text     = "Loading Spotify..."
        self.spotify_state    = "Stopped"

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
                "spotify_text":      self.spotify_text,
                "spotify_state":     self.spotify_state,
            }

class SpotifyPoller(threading.Thread):
    def __init__(self, shared: SharedState):
        super().__init__(daemon=True)
        self.shared = shared
    def run(self):
        while True:
            track, state = get_spotify_info()
            with self.shared.lock:
                self.shared.spotify_text = track
                self.shared.spotify_state = state
            time.sleep(1.0)


# ── Serial reader thread ──────────────────────────────────────────────────────

class LiveSerialReader(threading.Thread):
    """Background thread: reads ESP32 serial and fills shared.imu_buffer."""

    def __init__(self, port: str, baud: int, shared: SharedState):
        super().__init__(daemon=True)
        self.port   = port
        self.baud   = baud
        self.shared = shared
        self.fusion = Madgwick(sample_freq=100.0, beta=0.1)

    def run(self):
        while True:
            try:
                with serial.Serial(self.port, self.baud, timeout=1) as ser:
                    ser.flushInput()
                    with self.shared.lock:
                        self.shared.serial_connected = True
                    while True:
                        raw = ser.readline()
                        line = raw.decode("utf-8", errors="replace").strip()
                        parsed = parse_line(line)
                        if parsed:
                            accel = parsed[:3]
                            gyro_rad = [p * (math.pi/180.0) for p in parsed[3:6]]
                            self.fusion.update(accel, gyro_rad)
                            roll, pitch, yaw = self.fusion.get_euler()
                            row = list(parsed[:6]) + [roll, pitch, yaw, time.perf_counter()]
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
        wake_state = "IDLE"
        wait_start_time = 0.0

        while True:
            with self.shared.lock:
                buf = list(self.shared.imu_buffer)

            now = time.perf_counter()

            # ── Cooldown ──
            if wake_state == "IDLE" and (now - last_action_time < COOLDOWN):
                with self.shared.lock:
                    self.shared.imu_buffer.clear()
                    self.shared.status = "cooldown"
                time.sleep(0.05)
                continue

            if wake_state == "WAITING":
                with self.shared.lock:
                    self.shared.status = "detecting"
                    self.shared.last_action = "Waiting 3s..."
                if now - wait_start_time >= 3.0:
                    wake_state = "CAPTURING"
                    with self.shared.lock:
                        self.shared.imu_buffer.clear()
                        self.shared.last_action = "Recording command..."
                time.sleep(0.01)
                continue

            if len(buf) < 2:
                with self.shared.lock:
                    self.shared.status = "detecting"
                    if wake_state == "IDLE":
                        self.shared.last_action = "Listening for clap..."
                    elif wake_state == "CAPTURING":
                        self.shared.last_action = "Recording command..."
                time.sleep(0.01)
                continue

            buffer_duration = buf[-1][9] - buf[0][9]

            if buffer_duration < WINDOW_DURATION:
                with self.shared.lock:
                    self.shared.status = "detecting"
                    if wake_state == "IDLE":
                        self.shared.last_action = "Listening for clap..."
                    elif wake_state == "CAPTURING":
                        self.shared.last_action = "Recording command..."
                time.sleep(0.01)
                continue

            # ── Feature extraction (mirrors live_detect.py exactly) ──
            data_array = np.array(buf)[:, :9]
            features = []
            for col in range(9):
                axis_data = data_array[:, col]
                features.extend([
                    float(np.mean(axis_data)),
                    float(np.std(axis_data)),
                    float(np.max(axis_data)),
                    float(np.min(axis_data)),
                ])
                features.append(float(np.sum(axis_data)))
                features.append(float(np.sum(axis_data**2)))

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

            if wake_state == "IDLE":
                if prediction == "clap" and confidence > 0.35:
                    wake_state = "WAITING"
                    wait_start_time = now
                    with self.shared.lock:
                        self.shared.last_gesture = "clap"
                        self.shared.last_action = "Waiting 3s..."
                        self.shared.confidence_dict = conf_dict
                        self.shared.imu_buffer.clear()
                else:
                    slide = max(1, int(len(buf) * 0.2))
                    with self.shared.lock:
                        self.shared.last_gesture    = "idle"
                        self.shared.confidence_dict = conf_dict
                        self.shared.status          = "detecting"
                        self.shared.imu_buffer      = self.shared.imu_buffer[slide:]
            elif wake_state == "CAPTURING":
                if prediction != "idle":
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

                    execute_command(prediction)
                    last_action_time = time.perf_counter()
                    wake_state = "IDLE"
                else:
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
        self._init_spotify_widget()

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
            height_ratios=[1.8, 1.2, 2.0, 1.4, 1.2],
            hspace=0.55, wspace=0.32,
            left=0.05, right=0.97, top=0.95, bottom=0.02,
        )

        self.ax_instruction = self.fig.add_subplot(gs[0, :])
        self.ax_spotify     = self.fig.add_subplot(gs[1, 0])
        self.ax_detection   = self.fig.add_subplot(gs[1, 1])
        self.ax_confidence  = self.fig.add_subplot(gs[2, 0])
        self.ax_log         = self.fig.add_subplot(gs[2, 1])
        self.ax_waveform    = self.fig.add_subplot(gs[3, :])
        self.ax_health      = self.fig.add_subplot(gs[4, :])

        for ax in (self.ax_instruction, self.ax_detection, self.ax_confidence,
                   self.ax_log, self.ax_waveform, self.ax_health, self.ax_spotify):
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
        self.ax_spotify.set_title(
            "SPOTIFY NOW PLAYING", loc="left", fontsize=8,
            color=C["dim"], pad=4)

        for ax in (self.ax_instruction, self.ax_detection, self.ax_log, self.ax_health, self.ax_spotify):
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

        self._pipe_boxes = []
        self._pipe_texts = []
        labels = ["1. WAKE", "2. WAIT", "3. RECORD", "4. ACTION"]
        w = 0.22
        pad = 0.02
        for i, lbl in enumerate(labels):
            x = 0.03 + i * (w + pad)
            rect = FancyBboxPatch((x, 0.70), w, 0.20, boxstyle="round,pad=0.02", 
                                  facecolor=C["panel"], edgecolor=C["grid"], lw=1.5)
            ax.add_patch(rect)
            self._pipe_boxes.append(rect)
            txt = ax.text(x + w/2, 0.80, lbl, ha="center", va="center", color=C["dim"], fontsize=8, fontweight="bold")
            self._pipe_texts.append(txt)

        self._det_gesture = ax.text(
            0.5, 0.35, "—",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=22, fontweight="bold", color=C["dim"])

        self._det_action = ax.text(
            0.5, 0.10, "Listening for clap...",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color=C["dim"])

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
        self._update_spotify_widget(snap)

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
        status = snap["status"]
        last_action = snap["last_action"]
        gesture = snap["last_gesture"]
        conf = snap["confidence_dict"].get(gesture, 0.0)

        for b, t in zip(self._pipe_boxes, self._pipe_texts):
            b.set_facecolor(C["panel"])
            b.set_edgecolor(C["grid"])
            t.set_color(C["dim"])

        if "Listening for clap" in last_action:
            self._pipe_boxes[0].set_edgecolor(C["ax"])
            self._pipe_texts[0].set_color(C["ax"])
            self._det_gesture.set_text("WAITING FOR WAKE")
            self._det_gesture.set_color(C["dim"])
        elif "Waiting 3s" in last_action:
            self._pipe_boxes[1].set_facecolor("#3d2f16") # subtle amber
            self._pipe_boxes[1].set_edgecolor(C["amber"])
            self._pipe_texts[1].set_color(C["amber"])
            self._det_gesture.set_text("GET READY")
            self._det_gesture.set_color(C["amber"])
        elif "Recording command" in last_action:
            self._pipe_boxes[2].set_edgecolor(C["red"])
            self._pipe_texts[2].set_color(C["red"])
            self._det_gesture.set_text("RECORDING...")
            self._det_gesture.set_color(C["red"])
        else:
            self._pipe_boxes[3].set_facecolor("#1f3b28") # subtle green
            self._pipe_boxes[3].set_edgecolor(C["green"])
            self._pipe_texts[3].set_color(C["green"])
            disp = GESTURE_ACTIONS.get(gesture, (gesture,))[0]
            self._det_gesture.set_text(f"{disp.upper()} ({conf*100:.0f}%)")
            self._det_gesture.set_color(C["green"])

        self._det_action.set_text(last_action)

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

        t_now = buf[-1][9]
        t_lo  = t_now - WINDOW_DURATION - 1.0   # show 1 extra second of history

        rows = [r for r in buf if r[9] >= t_lo]
        if len(rows) < 2:
            return

        t_rel = np.array([r[9] - t_now for r in rows])  # 0 = now, negative = past
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

    def _init_spotify_widget(self):
        ax = self.ax_spotify
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        self._spotify_icon = ax.text(
            0.03, 0.5, "♫",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=24, color=C["green"])

        self._spotify_track = ax.text(
            0.08, 0.5, "Loading Spotify...",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=12, fontweight="bold", color=C["text"])

        self._spotify_status = ax.text(
            0.97, 0.5, "Stopped",
            transform=ax.transAxes, ha="right", va="center",
            fontsize=10, color=C["dim"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2a2d3a",
                      edgecolor=C["grid"], linewidth=1))

    def _update_spotify_widget(self, snap: dict):
        text = snap["spotify_text"]
        state = snap["spotify_state"]
        self._spotify_track.set_text(text)
        self._spotify_status.set_text(state.upper())
        if state.lower() == "playing":
            self._spotify_icon.set_color(C["green"])
            self._spotify_status.set_color(C["green"])
            self._spotify_status.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="#2a2d3a", edgecolor=C["green"], linewidth=1))
        else:
            self._spotify_icon.set_color(C["dim"])
            self._spotify_status.set_color(C["dim"])
            self._spotify_status.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="#2a2d3a", edgecolor=C["dim"], linewidth=1))

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
        SpotifyPoller(shared).start()
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