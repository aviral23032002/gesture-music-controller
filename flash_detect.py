import time
import joblib
import serial
import serial.tools.list_ports
import re
import numpy as np
from pynput.keyboard import Key, Controller
import os

# --- CONFIGURATION ---
BAUD_RATE = 115200
TRIGGER_G = 0.40      # Wake up threshold
WINDOW_SEC = 1.0      # Sharp 1s burst
COOLDOWN = 0.5        
LPF_ALPHA = 0.3
HPF_ALPHA = 0.95

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "flash_model.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "model", "flash_scaler.pkl")

keyboard = Controller()

LINE_RE = re.compile(
    r"AX:(?P<ax>[-\d.]+)\s+AY:(?P<ay>[-\d.]+)\s+AZ:(?P<az>[-\d.]+)"
    r"\s*\|\s*"
    r"GX:(?P<gx>[-\d.]+)\s+GY:(?P<gy>[-\d.]+)\s+GZ:(?P<gz>[-\d.]+)"
)

def find_port():
    ports = serial.tools.list_ports.comports()
    usb = [p for p in ports if "usb" in p.device.lower() or "serial" in p.device.lower()]
    return usb[0].device if usb else (ports[0].device if ports else None)

def main():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        print("Error: Flash model not trained. Run train_flash_model.py first.")
        return
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    port = find_port()
    
    # Filter state for 6 channels
    last_lp = np.zeros(6)
    last_hp = np.zeros(6)
    
    state = "ARMED"
    record_buffer = []
    last_action_time = 0

    print("--- FLASH-AI HYBRID STARTING (6-AXIS) ---")
    
    try:
        with serial.Serial(port, BAUD_RATE, timeout=0.01) as ser:
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line: continue
                m = LINE_RE.search(line)
                if not m: continue
                
                # 1. Band-Pass Filter (6 Channels)
                raw = np.array([
                    float(m.group("ax")), float(m.group("ay")), float(m.group("az")),
                    float(m.group("gx")), float(m.group("gy")), float(m.group("gz"))
                ])
                lp = last_lp * (1 - LPF_ALPHA) + raw * LPF_ALPHA
                hp = HPF_ALPHA * (last_hp + lp - last_lp)
                last_lp, last_hp = lp, hp
                
                # 2. State Machine
                if state == "ARMED":
                    # Instant Phys Trigger (Split sensitivity)
                    ax_mag = np.abs(hp[0])
                    az_mag = np.abs(hp[2])
                    
                    # More sensitive to lateral (X), less to vertical (Z)
                    triggered = (ax_mag > 0.25) or (az_mag > 0.40)
                    
                    if triggered and (time.time() - last_action_time) > COOLDOWN:
                        trigger_axis = 0 if ax_mag > 0.25 else 2
                        source = "X (Lateral)" if trigger_axis == 0 else "Z (Vertical)"
                        print(f"\n[*] {source} Trigger! Analyzing AI...", end='', flush=True)
                        state = "RECORDING"
                        record_start = time.time()
                        record_buffer = [hp]
                        raw_record_buffer = [raw]
                
                elif state == "RECORDING":
                    record_buffer.append(hp)
                    raw_record_buffer.append(raw)
                    if time.time() - record_start >= WINDOW_SEC:
                        # 3. AI Inference
                        # We need both raw and hp filtered for orientation + shape
                        data_hp = np.array(record_buffer)
                        data_raw = np.array(raw_record_buffer)
                        
                        features = []
                        for col in range(6):
                            data = data_hp[:, col]
                            
                            # Capture SIGNED pattern (No Unit Scaling)
                            chunks = np.array_split(data, 4)
                            for chunk in chunks:
                                if len(chunk) == 0:
                                    features.extend([0, 0, 0, 0])
                                else:
                                    features.extend([np.mean(chunk), np.std(chunk), np.max(chunk), np.min(chunk)])
                            
                            # Orientation and Intensity
                            features.extend([np.sum(data), np.max(np.abs(data)), np.mean(data_raw[:, col])])
                        
                        probs = model.predict_proba(scaler.transform([features]))[0]
                        max_idx = np.argmax(probs)
                        pred = model.classes_[max_idx]
                        conf = probs[max_idx]
                        
                        if pred != 'idle' and conf > 0.45:
                            # --- PHYSICS POLARITY DECISION ---
                            # Trust the AI for "YES ACTION", but trust Physics for "DIRECTION"
                            data_raw = np.array(raw_record_buffer)
                            
                            if trigger_axis == 0: # X-Axis (Lateral)
                                ax_data = data_raw[:, 0]
                                peak_val = ax_data[np.argmax(np.abs(ax_data))]
                                # Map sign to direction (Adjust polarities if needed)
                                actual_dir = "LEFT" if peak_val > 0 else "RIGHT" 
                                
                                print(f" -> {actual_dir} (AI Validated {conf*100:.0f}%)")
                                if actual_dir == "LEFT": keyboard.tap(Key.media_next)
                                else: keyboard.tap(Key.media_previous)
                            
                            else: # Z-Axis (Vertical)
                                az_data = data_raw[:, 2]
                                peak_val = az_data[np.argmax(np.abs(az_data))]
                                # Map sign to direction
                                actual_dir = "UP" if peak_val > 1.0 else "DOWN" # 1.0 is gravity baseline
                                
                                print(f" -> {actual_dir} (AI Validated {conf*100:.0f}%)")
                                if actual_dir == "UP": keyboard.tap(Key.media_volume_up)
                                else: keyboard.tap(Key.media_volume_down)
                        else:
                            # print(f" -> [REJECT] {pred} ({conf*100:.0f}%)")
                            pass
                            print(f" -> [REJECT] {pred} ({conf*100:.0f}%)")
                            
                        state = "ARMED"
                        last_action_time = time.time()
                        
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
