import math
import numpy as np
import joblib
from pynput.keyboard import Key, Controller
import time
import sys
import threading
import serial
import serial.tools.list_ports
import os
import re
import subprocess

# --- CONFIGURATION ---
BAUD_RATE = 115200
WINDOW_DURATION = 2.5 # matching the 2.5s window duration
COOLDOWN_DURATION = 2.0 # 2s wait before next gesture
CONFIDENCE_THRESHOLD = 0.50 
SAMPLE_FEQ = 100.0
MADGWICK_BETA = 0.1

# ANSI Colors
C_GREEN = "\033[92m"
C_CYAN = "\033[96m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"

LINE_RE = re.compile(
    r"AX:(?P<ax>[-\d.]+)\s+AY:(?P<ay>[-\d.]+)\s+AZ:(?P<az>[-\d.]+)"
    r"\s*\|\s*"
    r"GX:(?P<gx>[-\d.]+)\s+GY:(?P<gy>[-\d.]+)\s+GZ:(?P<gz>[-\d.]+)"
    r"\s*\|\s*"
    r"T:(?P<t>[-\d.]+)(?:\s*C)?" 
)

keyboard = Controller()

def find_port():
    ports = serial.tools.list_ports.comports()
    usb = [p for p in ports if "usb" in p.device.lower() or "serial" in p.device.lower()]
    return usb[0].device if usb else (ports[0].device if ports else None)

def send_spotify_command(cmd: str):
    try:
        subprocess.run(["osascript", "-e", f'tell application "Spotify" to {cmd}'], check=False)
    except Exception:
        pass

def execute_command(gesture):
    print(f"\n{C_BOLD}{C_GREEN}>>> GESTURE DETECTED: {gesture.upper()} <<<{C_RESET}")
    
    if gesture == 'up':
        send_spotify_command("set sound volume to (sound volume + 10)")
        print("Action: Volume Up 🔊")
    elif gesture == 'down':
        send_spotify_command("set sound volume to (sound volume - 10)")
        print("Action: Volume Down 🔉")
    elif gesture == 'right':
        send_spotify_command("next track")
        print("Action: Next Track ⏭️")
    elif gesture == 'left':
        send_spotify_command("previous track")
        print("Action: Previous Track ⏮️")
    elif gesture == 'clap':
        # Clap is our wake word, no direct media action
        pass
    elif gesture == 'push':
        send_spotify_command("playpause")
        print("Action: Play / Pause ⏯️")
    elif gesture == 'pull':
        send_spotify_command("playpause")
        print("Action: Play / Pause ⏯️")
    elif gesture == 'wrist_rotate_right':
        send_spotify_command("set player position to (player position + 10)")
        print("Action: Fast Forward +10s ⏩")
    elif gesture == 'wrist_rotate_left':
        send_spotify_command("set player position to (player position - 10)")
        print("Action: Rewind -10s ⏪")
    elif gesture == 'idle':
        print("Action: Idle detected (No action)")
    else:
        print(f"Unknown gesture: {gesture}")

class Madgwick:
    def __init__(self, sample_freq=100.0, beta=0.1):
        self.sample_freq = sample_freq
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, accel, gyro):
        gx, gy, gz = gyro
        ax, ay, az = accel
        q = self.q
        norm = np.linalg.norm(accel)
        if norm == 0: return
        ax /= norm; ay /= norm; az /= norm
        f = np.array([
            2.0*(q[1]*q[3] - q[0]*q[2]) - ax,
            2.0*(q[0]*q[1] + q[2]*q[3]) - ay,
            2.0*(0.5 - q[1]**2 - q[2]**2) - az
        ])
        j = np.array([
            [-2.0*q[2], 2.0*q[3], -2.0*q[0], 2.0*q[1]],
            [ 2.0*q[1], 2.0*q[0],  2.0*q[3], 2.0*q[2]],
            [ 0.0,     -4.0*q[1], -4.0*q[2], 0.0]
        ])
        step = j.T @ f
        step_norm = np.linalg.norm(step)
        if step_norm > 0: step /= step_norm
        else: step = np.zeros(4)
        q_dot = 0.5 * self.q_mult(q, [0, gx, gy, gz]) - self.beta * step
        self.q += q_dot * (1.0 / self.sample_freq)
        self.q /= np.linalg.norm(self.q)

    def q_mult(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def get_euler(self):
        w, x, y, z = self.q
        roll = math.atan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
        pitch = math.asin(max(-1.0, min(1.0, 2.0*(w*y - z*x))))
        yaw = math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
        return roll, pitch, yaw

class LiveSerialReader(threading.Thread):
    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.buffer = []
        self.lock = threading.Lock()
        self.running = True
        self.fusion = Madgwick(sample_freq=SAMPLE_FEQ, beta=MADGWICK_BETA)

    def run(self):
        try:
            with serial.Serial(self.port, self.baud, timeout=0.1) as ser:
                ser.flushInput()
                while self.running:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line: continue
                    m = LINE_RE.search(line)
                    if m:
                        parsed = tuple(float(m.group(k)) for k in ("ax", "ay", "az", "gx", "gy", "gz", "t"))
                        accel = parsed[:3]
                        gyro_rad = [p * (math.pi/180.0) for p in parsed[3:6]]
                        current_time = time.time()
                        with self.lock:
                            self.fusion.update(accel, gyro_rad)
                            roll, pitch, yaw = self.fusion.get_euler()
                            self.buffer.append(list(parsed[:6]) + [roll, pitch, yaw, current_time])
                            if len(self.buffer) > 1000: self.buffer.pop(0)
        except Exception as e:
            print(f"\n{C_RED}Serial error: {e}{C_RESET}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model", "gesture_model_2.5sec.pkl")
    scaler_path = os.path.join(script_dir, "model", "gesture_scaler_2.5sec.pkl")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print(f"{C_RED}Error: Trained 2.5s model not found inside 'model/' directory.{C_RESET}")
        print("Please run train_model.py first to train and generate the model files.")
        return
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    port = find_port()
    
    print(f"\n{C_BOLD}{C_CYAN}=== 8-GESTURE-CONTROLLER LIVE DETECTOR ==={C_RESET}")
    if not port:
        print(f"{C_RED}Error: No serial port found. Connect your device.{C_RESET}"); return
    print(f"Connected to device on: {C_GREEN}{port}{C_RESET}")
        
    reader = LiveSerialReader(port, BAUD_RATE)
    reader.start()

    # Workflow State: CALIBRATING, ARMED, CAPTURING_WAKE, WAITING, CAPTURING_ACTION, COOLDOWN
    system_status = "CALIBRATING"
    baseline_rms = 0.05
    calibration_samples = []
    capture_start_time = 0
    wait_start_time = 0
    cooldown_start_time = 0

    print(f"\n{C_CYAN}--- Beginning Calibration (Hold sensor still) ---{C_RESET}")
    try:
        while True:
            with reader.lock:
                current_buffer = list(reader.buffer)
            if not current_buffer:
                time.sleep(0.1); continue

            recent_data = np.array(current_buffer[-10:])[:, :3] if len(current_buffer) >= 10 else np.array(current_buffer)[:, :3]
            recent_detrended = recent_data - np.mean(recent_data, axis=0)
            rms = np.sqrt(np.mean(np.sum(recent_detrended**2, axis=1)))
            
            # Dynamic Calibration Logic
            dynamic_threshold = max(baseline_rms * 1.3, 0.015) 
            is_moving = rms > dynamic_threshold
            
            motion_tag = f"{C_RED}[MOVING]{C_RESET}" if is_moving else f"{C_GREEN}[STILL]{C_RESET}"

            if system_status == "CALIBRATING":
                if len(calibration_samples) < 10:
                    with reader.lock: reader.buffer = []
                
                calibration_samples.append(rms)
                print(f"\rStep 1: CALIBRATING... {len(calibration_samples)}/100 {motion_tag} (RMS: {rms:.4f}g)   ", end='', flush=True)
                if len(calibration_samples) >= 100:
                    baseline_rms = np.mean(calibration_samples[20:])
                    print(f"\n{C_GREEN}[CALIBRATION COMPLETE] Noise Floor: {baseline_rms:.4f}g{C_RESET}")
                    system_status = "ARMED"
                continue

            if system_status == "ARMED":
                print(f"\r[*] Listening for CLAP... Activity: {rms:.3f}g / Trigger: {dynamic_threshold:.3f}g      ", end='', flush=True)
                if is_moving:
                    print(f"\n{C_YELLOW}[CAPTURING] Motion detected, checking for clap...{C_RESET}")
                    capture_start_time = time.time()
                    system_status = "CAPTURING_WAKE"
                    with reader.lock:
                        reader.fusion = Madgwick(sample_freq=SAMPLE_FEQ, beta=MADGWICK_BETA)
                        reader.buffer = []
                continue

            if system_status == "CAPTURING_WAKE":
                duration = time.time() - capture_start_time
                if duration >= WINDOW_DURATION:
                    if len(current_buffer) < 50:
                        print(f"\n{C_RED}[WEAK] Capture too short. Rearming...{C_RESET}")
                        system_status = "ARMED"
                        continue
                        
                    data_array = np.array(current_buffer)[:, :9]
                    features = []
                    for col in range(9):
                        axis_data = data_array[:, col]
                        features.extend([np.mean(axis_data), np.std(axis_data), np.max(axis_data), np.min(axis_data)])
                        features.append(np.sum(axis_data))
                        features.append(np.sum(axis_data**2))
                    
                    probs = model.predict_proba(scaler.transform([features]))[0]
                    max_idx = np.argmax(probs)
                    prediction = model.classes_[max_idx]
                    confidence = probs[max_idx]
                    
                    if prediction == 'clap' and confidence >= 0.35:
                        print(f"{C_GREEN}[WAKE WORD] Clap detected! ({confidence * 100:.1f}%){C_RESET}")
                        print(f"{C_CYAN}[WAITING] Waiting 3 seconds for next gesture...{C_RESET}")
                        wait_start_time = time.time()
                        system_status = "WAITING"
                    else:
                        print(f"\n{C_YELLOW}[IGNORED] Detected {prediction} ({confidence*100:.1f}%), waiting for CLAP.{C_RESET}")
                        system_status = "ARMED"
                continue

            if system_status == "WAITING":
                if time.time() - wait_start_time >= 3.0:
                    print(f"\n{C_YELLOW}[CAPTURING ACTION] Recording command gesture now...{C_RESET}")
                    capture_start_time = time.time()
                    system_status = "CAPTURING_ACTION"
                    with reader.lock:
                        reader.fusion = Madgwick(sample_freq=SAMPLE_FEQ, beta=MADGWICK_BETA)
                        reader.buffer = []
                continue

            if system_status == "CAPTURING_ACTION":
                duration = time.time() - capture_start_time
                if duration >= WINDOW_DURATION:
                    if len(current_buffer) < 50:
                        print(f"\n{C_RED}[WEAK] Capture too short. Rearming...{C_RESET}")
                        system_status = "ARMED"
                        continue
                        
                    data_array = np.array(current_buffer)[:, :9]
                    features = []
                    for col in range(9):
                        axis_data = data_array[:, col]
                        features.extend([np.mean(axis_data), np.std(axis_data), np.max(axis_data), np.min(axis_data)])
                        features.append(np.sum(axis_data))
                        features.append(np.sum(axis_data**2))
                    
                    probs = model.predict_proba(scaler.transform([features]))[0]
                    max_idx = np.argmax(probs)
                    prediction = model.classes_[max_idx]
                    confidence = probs[max_idx]
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        execute_command(prediction)
                        print(f"Confidence: {C_BOLD}{confidence * 100:.1f}%{C_RESET}")
                    else:
                        print(f"\n{C_YELLOW}[WEAK] Guess: {prediction} ({confidence:.2f}){C_RESET}")
                    
                    print(f"[COOLDOWN] Waiting {COOLDOWN_DURATION}s...")
                    cooldown_start_time = time.time()
                    system_status = "COOLDOWN"
                continue

            if system_status == "COOLDOWN":
                elapsed = time.time() - cooldown_start_time
                remaining = COOLDOWN_DURATION - elapsed
                if remaining > 0:
                    print(f"\r[WAIT] Cooldown: {remaining:.1f}s... {motion_tag}         ", end='', flush=True)
                else:
                    print(f"\n{C_GREEN}[READY] Armed and listening for CLAP...{C_RESET}")
                    system_status = "ARMED"
                continue
            time.sleep(0.01)
    except KeyboardInterrupt:
        print(f"\n\n{C_RED}Exiting Live Detector...{C_RESET}\n")
        reader.running = False
        reader.join()

if __name__ == "__main__":
    main()
