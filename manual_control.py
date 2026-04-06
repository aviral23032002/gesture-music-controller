import os
import re
import time
import sys
import threading
import serial
import serial.tools.list_ports
import numpy as np
import joblib
from pynput.keyboard import Key, Controller

# --- CONFIGURATION ---
BAUD_RATE = 115200
WINDOW_DURATION = 2.0  

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
    if gesture == 'up':
        keyboard.tap(Key.media_volume_up)
        print("   -> Action: System Volume Up")
    elif gesture == 'down':
        keyboard.tap(Key.media_volume_down)
        print("   -> Action: System Volume Down")
    elif gesture == 'rotate_right':
        keyboard.tap(Key.media_next)
        print("   -> Action: Global Next Track")
    elif gesture == 'rotate_left':
        keyboard.tap(Key.media_previous)
        print("   -> Action: Global Previous Track")
    elif gesture == 'shake':
        keyboard.tap(Key.media_play_pause)
        print("   -> Action: Global Play/Pause")
    elif gesture == 'push':
        keyboard.tap(Key.media_volume_mute)
        print("   -> Action: Global Volume Mute")

# --- Background Thread ---
class LiveSerialReader(threading.Thread):
    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.lock = threading.Lock()
        self.buffer = []
        self.running = True

    def run(self):
        try:
            with serial.Serial(self.port, self.baud, timeout=1) as ser:
                while self.running:
                    raw = ser.readline()
                    decoded_line = raw.decode("utf-8", errors="replace").strip()
                    parsed = parse_line(decoded_line)
                    
                    if parsed:
                        with self.lock:
                            self.buffer.append(parsed[:6]) # Only save the 6 sensor axes
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
        
    print(f"Connecting to {port}...\n")

    reader = LiveSerialReader(port, BAUD_RATE)
    reader.start()
    
    # Wait a moment for serial to stabilize
    time.sleep(1)

    try:
        while True:
            # 1. Wait for user trigger
            user_input = input("\nPress ENTER to perform a gesture (or type 'q' to quit)...")
            
            if user_input.strip().lower() == 'q':
                break
                
            # 2. Clear out any old data from the background thread
            with reader.lock:
                reader.buffer.clear()
                
            # 3. Record for exactly the window duration
            print(f"[*] RECORDING FOR {WINDOW_DURATION} SECONDS! Do your gesture NOW...")
            time.sleep(WINDOW_DURATION)
            
            # 4. Grab the recorded data
            with reader.lock:
                current_buffer = list(reader.buffer)
                
            print("[*] Recording stopped. Analyzing...")
            
            # 5. Extract Features and Predict
            if len(current_buffer) > 10: # Ensure we actually captured data
                data_array = np.array(current_buffer)
                features = []
                for col in range(6):
                    axis_data = data_array[:, col]
                    features.extend([
                        np.mean(axis_data), 
                        np.std(axis_data), 
                        np.max(axis_data), 
                        np.min(axis_data)
                    ])
                
                features_scaled = scaler.transform([features])
                prediction = model.predict(features_scaled)[0]
                
                print(f"\n>>> MODEL DETECTED: {prediction.upper()} <<<")
                
                if prediction != 'idle':
                    execute_command(prediction)
            else:
                print("\n[!] Not enough data captured. Make sure the ESP32 is transmitting.")

    except KeyboardInterrupt:
        pass
        
    finally:
        print("\n\nExiting manual control...")
        reader.running = False
        sys.exit()

if __name__ == "__main__":
    main()