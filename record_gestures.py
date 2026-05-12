import os
import re
import time
import sys
import threading
import serial
import serial.tools.list_ports
import pandas as pd

# --- CONFIGURATION ---
BAUD_RATE = 115200
DURATION = 2.5          # Set to 2.5 seconds
NUM_SAMPLES = 25      # Set to 25 samples
CURRENT_GESTURE = 'wrist_rotate_left' # Change this manually (e.g., 'down', 'rotate_right', 'idle')

LINE_RE = re.compile(
    r"AX:(?P<ax>[-\d.]+)\s+AY:(?P<ay>[-\d.]+)\s+AZ:(?P<az>[-\d.]+)"
    r"\s*\|\s*"
    r"GX:(?P<gx>[-\d.]+)\s+GY:(?P<gy>[-\d.]+)\s+GZ:(?P<gz>[-\d.]+)"
    r"\s*\|\s*"
    r"T:(?P<t>[-\d.]+)"
)

def find_port():
    """Finds the connected Arduino/IMU serial port."""
    ports = serial.tools.list_ports.comports()
    usb = [p for p in ports if "usb" in p.device.lower() or "serial" in p.device.lower()]
    return usb[0].device if usb else (ports[0].device if ports else None)

def parse_line(line: str):
    """Parses a single line of serial data based on the Regex."""
    m = LINE_RE.search(line)
    if m:
        return tuple(float(m.group(k)) for k in ("ax", "ay", "az", "gx", "gy", "gz", "t"))
    return None

class SerialReader(threading.Thread):
    """Reads serial data in the background so it doesn't block the main thread."""
    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.lock = threading.Lock()
        self.t, self.ax, self.ay, self.az = [], [], [], []
        self.gx, self.gy, self.gz = [], [], []
        self.running = True

    def run(self):
        try:
            with serial.Serial(self.port, self.baud, timeout=1) as ser:
                t0 = time.perf_counter()
                while self.running:
                    raw = ser.readline()
                    parsed = parse_line(raw.decode("utf-8", errors="replace").strip())
                    if parsed:
                        now = time.perf_counter() - t0
                        with self.lock:
                            self.t.append(now)
                            self.ax.append(parsed[0]); self.ay.append(parsed[1]); self.az.append(parsed[2])
                            self.gx.append(parsed[3]); self.gy.append(parsed[4]); self.gz.append(parsed[5])
        except Exception as e: 
            print(f"Serial Error: {e}")

def main():
    # --- DIRECTORY SETUP ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(script_dir, "data")
    
    # Create the specific folder for the CURRENT_GESTURE
    gesture_dir = os.path.join(base_data_dir, CURRENT_GESTURE)
    os.makedirs(gesture_dir, exist_ok=True)
    
    print(f"Data will be saved to: {gesture_dir}\n")

    port = find_port()
    if not port:
        print("Error: No serial port found. Check your device connection.")
        sys.exit()
    print(f"Connected to device on port: {port}\n")

    try:
        print(f"=== Get ready to record '{CURRENT_GESTURE.upper()}' ===")
        print(f"Target: {NUM_SAMPLES} samples, {DURATION} seconds each.\n")
        
        for i in range(NUM_SAMPLES):
            # Wait for user input before EVERY recording
            input(f"[{CURRENT_GESTURE.upper()} {i+1}/{NUM_SAMPLES}] Press ENTER to start recording...")
            
            filename = f"{CURRENT_GESTURE}_{i:02d}.txt"
            
            # Save to the specific gesture directory
            filepath = os.path.join(gesture_dir, filename)

            print(f"--- RECORDING {filename} NOW ({DURATION}s) ---")
            reader = SerialReader(port, BAUD_RATE)
            reader.start()
            
            time.sleep(DURATION)
            
            reader.running = False
            reader.join()
            print("--- Stopped ---")

            with reader.lock:
                df = pd.DataFrame({
                    'Time': reader.t,
                    'AX': reader.ax, 'AY': reader.ay, 'AZ': reader.az,
                    'GX': reader.gx, 'GY': reader.gy, 'GZ': reader.gz
                })
            
            # Save the recorded data
            df.to_csv(filepath, index=False, sep=' ')
        
        print(f"\n*** Finished all {NUM_SAMPLES} samples for '{CURRENT_GESTURE.upper()}' ***\n")

    except KeyboardInterrupt:
        print("\n\nRecording sequence interrupted by user. Exiting safely...")
        sys.exit()

    print("\nData collection for this session is complete!")

if __name__ == "__main__":
    main()