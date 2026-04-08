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
WINDOW_DURATION = 2.5    # 2.5s is the optimal human gesture window
WINDOW_OVERLAP = 0.8     # checks every 0.5 seconds
CONFIDENCE_THRESHOLD = 0.70 # Balanced stability
DEBOUNCE_TIME = 1.0
CONSECUTIVE_WINDOWS = 1  
CONTINUOUS_GESTURES = ['up', 'down'] # volume gestures

# Sensor Fusion Constants
SAMPLE_FEQ = 100.0       # 100 Hz from main.c
MADGWICK_BETA = 0.1      # Filter gain

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

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

def parse_line(line: str):
    m = LINE_RE.search(line)
    if m:
        return tuple(float(m.group(k)) for k in ("ax", "ay", "az", "gx", "gy", "gz", "t"))
    return None

def execute_command(gesture):
    print(f"\n>>> EXECUTING: {gesture.upper()} <<<")
    
    if gesture == 'up':
        # Taps the global volume up key
        keyboard.tap(Key.media_volume_up)
        print("Action: System Volume Up")
        
    elif gesture == 'down':
        # Taps the global volume down key
        keyboard.tap(Key.media_volume_down)
        print("Action: System Volume Down")
        
    elif gesture == 'rotate_right':
        # Taps the global "Next Track" media key
        keyboard.tap(Key.media_next)
        print("Action: Global Next Track")
        
    elif gesture == 'rotate_left':
        # Taps the global "Previous Track" media key
        keyboard.tap(Key.media_previous)
        print("Action: Global Previous Track")
    
    elif gesture == 'shake':
        # Taps the global "Play/Pause" media key
        keyboard.tap(Key.media_play_pause)
        print("Action: Global Play/Pause")
    
    elif gesture == 'push':
        # Taps the global "Volume Mute" media key
        keyboard.tap(Key.media_volume_mute)
        print("Action: Global Volume Mute")
        
    else:
        print("Unknown command. Try 'up', 'down', 'rotate_right', or 'rotate_left'.")

class Madgwick:
    """Madgwick AHRS implementation for sensor fusion."""
    def __init__(self, sample_freq=100.0, beta=0.1):
        self.sample_freq = sample_freq
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) # Quaternion [w, x, y, z]

    def update(self, accel, gyro):
        # Gyro should be in rad/s, Accel in g
        gx, gy, gz = gyro
        ax, ay, az = accel

        q = self.q

        # Normalise accelerometer measurement
        norm = np.linalg.norm(accel)
        if norm == 0: return
        ax /= norm; ay /= norm; az /= norm

        # Gradient decent algorithm corrective step
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
        if step_norm > 0:
            step /= step_norm
        else:
            step = np.zeros(4)

        # Compute rate of change of quaternion
        q_dot = 0.5 * self.q_mult(q, [0, gx, gy, gz]) - self.beta * step

        # Integrate to yield quaternion
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

def compute_activity(data_array):
    """Computes RMS of accelerometer data after removing the mean (gravity removal)."""
    # First 3 columns are AX, AY, AZ
    accel = data_array[:, :3]
    # Subtract mean from each axis (crude gravity removal)
    accel_detrended = accel - np.mean(accel, axis=0)
    # RMS = sqrt(mean(x^2 + y^2 + z^2))
    rms = np.sqrt(np.mean(np.sum(accel_detrended**2, axis=1)))
    return rms

# --- NEW: Background Thread from your recording script ---
class LiveSerialReader(threading.Thread):
    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.lock = threading.Lock()
        self.buffer = []
        self.running = True
        self.fusion = Madgwick(sample_freq=SAMPLE_FEQ, beta=MADGWICK_BETA)

    def run(self):
        try:
            with serial.Serial(self.port, self.baud, timeout=1) as ser:
                while self.running:
                    raw = ser.readline()
                    decoded_line = raw.decode("utf-8", errors="replace").strip()
                    parsed = parse_line(decoded_line)
                    
                    if parsed:
                        # Raw data: [ax, ay, az, gx, gy, gz]
                        accel = np.array(parsed[:3])
                        gyro_deg = np.array(parsed[3:6])
                        gyro_rad = gyro_deg * (math.pi / 180.0)
                        
                        # Update Madgwick Filter
                        self.fusion.update(accel, gyro_rad)
                        roll, pitch, yaw = self.fusion.get_euler()
                        
                        current_time = time.perf_counter()
                        # Data row: [ax, ay, az, gx, gy, gz, roll, pitch, yaw, time]
                        data_row = list(parsed[:6]) + [roll, pitch, yaw, current_time]
                        
                        with self.lock:
                            self.buffer.append(data_row)
                    else:
                        # Log non-matching lines occasionally to help debugging
                        if time.time() % 5 < 0.01:
                            print(f"\n[DEBUG] Raw sample (not matching regex): {decoded_line}")
        except Exception as e:
            print(f"\nSerial Error: {e}")

def main():
    model_path = os.path.join(MODEL_DIR, 'gesture_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'gesture_scaler.pkl')

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"Model and scaler loaded successfully from {MODEL_DIR}.")
    except Exception as e:
        print(f"Error loading model files. Did you run train_model.py first?")
        sys.exit()

    port = find_port()
    if not port:
        print("Error: No serial port found.")
        sys.exit()
        
    print(f"Connecting to {port}...\nListening for gestures! (Press Ctrl+C to quit)\n", flush=True)

    reader = LiveSerialReader(port, BAUD_RATE)
    reader.start()
    
    # Workflow State: CALIBRATING, READY, or LOCKOUT
    system_status = "CALIBRATING"
    baseline_rms = 0.05 # default
    calibration_samples = []
    
    print("\n--- SYSTEM STARTING ---")
    print("Step 1: CALIBRATING. Please leave the sensor COMPLETELY STILL for 2 seconds...")

    try:
        while True:
            # 1. Capture current buffer
            with reader.lock:
                current_buffer = list(reader.buffer)
            
            if len(current_buffer) == 0:
                time.sleep(0.1)
                continue
            
            # 2. Check current motion (using last 10 samples for speed)
            recent_data = np.array(current_buffer[-10:])[:, :3] if len(current_buffer) >= 10 else np.array(current_buffer)[:, :3]
            recent_detrended = recent_data - np.mean(recent_data, axis=0)
            rms = np.sqrt(np.mean(np.sum(recent_detrended**2, axis=1)))
            
            # Dynamic Threshold Calculation
            dynamic_threshold = baseline_rms * 2.0 # Allow 2x the noise floor
            is_moving = rms > dynamic_threshold

            # --- WORKFLOW STEP 0: Calibration ---
            if system_status == "CALIBRATING":
                calibration_samples.append(rms)
                if len(calibration_samples) > 100: # ~1 second of data
                    baseline_rms = np.mean(calibration_samples)
                    print(f"[CALIB] Noise floor detected at: {baseline_rms:.4f}g")
                    system_status = "LOCKOUT" # Transition to wait-for-idle
                continue

            # --- WORKFLOW STEP 4: Reset from Lockout to Ready ---
            if system_status == "LOCKOUT":
                if not is_moving:
                    system_status = "READY"
                    print(f"\n[READY] Watching for gestures (Threshold: {dynamic_threshold:.3f}g)")
                    with reader.lock:
                        reader.fusion = Madgwick(sample_freq=SAMPLE_FEQ, beta=MADGWICK_BETA)
                        reader.buffer = []
                continue

            # --- WORKFLOW STEP 2: Detect in 4s windows ---
            buffer_duration = current_buffer[-1][9] - current_buffer[0][9]
            if buffer_duration >= WINDOW_DURATION:
                # CRITICAL FIX: Extract EXACTLY WINDOW_DURATION worth of data
                # Otherwise features are calculated over the entire history!
                end_time_threshold = current_buffer[0][9] + WINDOW_DURATION
                window_slice = []
                for row in current_buffer:
                    if row[9] <= end_time_threshold:
                        window_slice.append(row)
                    else:
                        break
                
                data_array = np.array(window_slice)[:, :9]

                # We only predict if we are currently mid-motion
                if is_moving:
                    features = []
                    for col in range(9):
                        axis_data = data_array[:, col]
                        features.extend([np.mean(axis_data), np.std(axis_data), np.max(axis_data), np.min(axis_data)])
                    
                    try:
                        features_scaled = scaler.transform([features])
                        probs = model.predict_proba(features_scaled)[0]
                        max_idx = np.argmax(probs)
                        prediction = model.classes_[max_idx]
                        confidence = probs[max_idx]
                        
                        # Diagnostic: Show what the model thinks (silent unless you move)
                        if is_moving:
                             print(f"[*] Thinking... {prediction.ljust(12)} (Conf: {confidence:.2f})       ", end='\r', flush=True)
                        
                        # --- WORKFLOW STEP 3: Take Action Immediately ---
                        if prediction not in ['null', 'idle'] and confidence >= CONFIDENCE_THRESHOLD:
                            execute_command(prediction)
                            print(f"(Confidence: {confidence:.2f})")
                            system_status = "LOCKOUT"
                            print("[LOCKOUT] Action fired. Please return to Idle to reset.")
                            # Clear buffer so we don't double-trigger
                            with reader.lock:
                                reader.buffer = []
                    except Exception as e:
                        print(f"\n[ERROR] Prediction failed: {e}")
                        sys.exit(1)

                # Slide the buffer if we haven't fired yet
                if system_status == "READY":
                    slide_step = WINDOW_DURATION * (1 - WINDOW_OVERLAP)
                    slide_time_limit = current_buffer[0][9] + slide_step
                    drop_count = 0
                    for row in current_buffer:
                        if row[9] < slide_time_limit:
                            drop_count += 1
                        else:
                            break
                    with reader.lock:
                        reader.buffer = reader.buffer[drop_count:]
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nExiting live control...")
        reader.running = False
        sys.exit()

if __name__ == "__main__":
    main()