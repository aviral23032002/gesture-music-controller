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

# --- CONFIGURATION ---
BAUD_RATE = 115200
WINDOW_DURATION = 2.0 # Fast capture
COOLDOWN_DURATION = 2.0 # 2s wait before next
WINDOW_OVERLAP = 0.8    
CONFIDENCE_THRESHOLD = 0.50 
SAMPLE_FEQ = 100.0
MADGWICK_BETA = 0.1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "gesture_model_2.5sec.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "model", "gesture_scaler_2.5sec.pkl")

keyboard = Controller()

LINE_RE = re.compile(
    r"AX:(?P<ax>[-\d.]+)\s+AY:(?P<ay>[-\d.]+)\s+AZ:(?P<az>[-\d.]+)"
    r"\s*\|\s*"
    r"GX:(?P<gx>[-\d.]+)\s+GY:(?P<gy>[-\d.]+)\s+GZ:(?P<gz>[-\d.]+)"
    r"\s*\|\s*"
    r"T:(?P<t>[-\d.]+)(?:\s*C)?" 
)

def find_port():
    ports = serial.tools.list_ports.comports()
    usb = [p for p in ports if "usb" in p.device.lower() or "serial" in p.device.lower()]
    return usb[0].device if usb else (ports[0].device if ports else None)

def execute_command(gesture):
    print(f"\n>>> EXECUTING: {gesture.upper()} <<<")
    
    if gesture == 'up':
        keyboard.tap(Key.media_volume_up)
        print("Action: System Volume Up")
    elif gesture == 'down':
        keyboard.tap(Key.media_volume_down)
        print("Action: System Volume Down")
    elif gesture == 'right':
        keyboard.tap(Key.media_next)
        print("Action: Global Next Track")
    elif gesture == 'left':
        keyboard.tap(Key.media_previous)
        print("Action: Global Previous Track")
    else:
        print(f"Unknown command: {gesture}")

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
            print(f"Serial error: {e}")

def main():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        print("Error: 2.5sec model not found. Run train_model_2.5sec.py first.")
        return
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    port = find_port()
    if not port:
        print("No serial port found."); return
        
    reader = LiveSerialReader(port, BAUD_RATE)
    reader.start()

    # Workflow State: CALIBRATING, ARMED, CAPTURING, COOLDOWN
    system_status = "CALIBRATING"
    baseline_rms = 0.05
    calibration_samples = []
    capture_start_time = 0
    cooldown_start_time = 0

    print("\n--- 2s TRIGGER | 2s COOLDOWN STARTING ---")
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
            
            # Helper for display
            motion_tag = "[MOVING]" if is_moving else "[STILL ]"

            if system_status == "CALIBRATING":
                # Flush initial jitter
                if len(calibration_samples) < 10:
                    with reader.lock: reader.buffer = []
                
                calibration_samples.append(rms)
                print(f"\rStep 1: CALIBRATING... {len(calibration_samples)}/100 {motion_tag} (RMS: {rms:.4f}g)", end='', flush=True)
                if len(calibration_samples) >= 100:
                    baseline_rms = np.mean(calibration_samples[20:]) # Ignore first 20 samples
                    print(f"\n[CALIB DONE] Noise Floor: {baseline_rms:.4f}g")
                    system_status = "ARMED"
                continue

            if system_status == "READY":
                if not is_moving:
                    print(f"\r[READY] Armed. Trigger: >{dynamic_threshold:.3f}g  {motion_tag}   ", end='', flush=True)
                    system_status = "ARMED"
                continue

            if system_status == "ARMED":
                # Debug: Show current motion level
                print(f"\r[*] Listening... Activity: {rms:.3f}g / {dynamic_threshold:.3f}g      ", end='', flush=True)
                if is_moving:
                    print("\n[CAPTURING] Recording 2.5s gesture...")
                    capture_start_time = time.time()
                    system_status = "CAPTURING"
                    with reader.lock:
                        reader.fusion = Madgwick(sample_freq=SAMPLE_FEQ, beta=MADGWICK_BETA)
                        reader.buffer = []
                continue

            if system_status == "CAPTURING":
                duration = time.time() - capture_start_time
                if duration >= WINDOW_DURATION:
                    data_array = np.array(current_buffer)[:, :9]
                    features = []
                    for col in range(9):
                        axis_data = data_array[:, col]
                        # Stats
                        features.extend([np.mean(axis_data), np.std(axis_data), np.max(axis_data), np.min(axis_data)])
                        # Energy/Impulse
                        features.append(np.sum(axis_data))
                        features.append(np.sum(axis_data**2))
                    
                    probs = model.predict_proba(scaler.transform([features]))[0]
                    max_idx = np.argmax(probs)
                    prediction = model.classes_[max_idx]
                    confidence = probs[max_idx]
                    
                    if prediction not in ['idle', 'null'] and confidence >= CONFIDENCE_THRESHOLD:
                        execute_command(prediction)
                        print(f"(Conf: {confidence:.2f})")
                    else:
                        print(f"\n[WEAK] Guess: {prediction} ({confidence:.2f})")
                    
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
                    print("\n[READY] Armed. Move now! ")
                    system_status = "ARMED"
                continue
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nExiting..."); reader.running = False
if __name__ == "__main__":
    main()
