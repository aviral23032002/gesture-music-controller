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
import time
import sys

# --- CONFIGURATION ---
BAUD_RATE = 115200
WINDOW_DURATION = 2.0  
COOLDOWN = 1.5         

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

# --- NEW: Background Thread from your recording script ---
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
                        current_time = time.perf_counter()
                        data_row = list(parsed[:6]) + [current_time]
                        with self.lock:
                            self.buffer.append(data_row)
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
        
    print(f"Connecting to {port}...\nListening for gestures! (Press Ctrl+C to quit)\n")

    # Start the background reader thread
    reader = LiveSerialReader(port, BAUD_RATE)
    reader.start()
    
    last_action_time = 0

    try:
        while True:
            # Safely copy the buffer from the background thread
            with reader.lock:
                current_buffer = list(reader.buffer)
            
            now = time.perf_counter()
            
            # If we are in the cooldown period, keep clearing the buffer so we don't hold stale data
            if now - last_action_time < COOLDOWN:
                with reader.lock:
                    reader.buffer.clear()
                time.sleep(0.05)
                continue
                
            if len(current_buffer) > 0:
                buffer_duration = current_buffer[-1][6] - current_buffer[0][6]
                
                if buffer_duration >= WINDOW_DURATION:
                    data_array = np.array(current_buffer)[:, :6] 
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
                    
                    print(f"Current State: {prediction.upper().ljust(15)} | Buffer size: {len(current_buffer)}", end='\r')
                    
                    if prediction != 'idle':
                        execute_command(prediction)
                        last_action_time = time.perf_counter()
                        with reader.lock:
                            reader.buffer.clear()
                    else:
                        slide_amount = int(len(current_buffer) * 0.2)
                        with reader.lock:
                            reader.buffer = reader.buffer[slide_amount:]
            
            # Small sleep to prevent the main thread from maxing out the CPU
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nExiting live control...")
        reader.running = False
        sys.exit()

if __name__ == "__main__":
    main()