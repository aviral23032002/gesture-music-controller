import serial
import time
import os
import argparse
import numpy as np
import re

# --- CONFIGURATION ---
BAUD_RATE = 115200
TRIGGER_G = 0.40
WINDOW_SEC = 1.0
SAMPLES_NEEDED = 20

DATA_DIR = "data_flash"
LINE_RE = re.compile(r"AX:(?P<ax>[-\d.]+)\s+AY:(?P<ay>[-\d.]+)\s+AZ:(?P<az>[-\d.]+)\s*\|\s*GX:(?P<gx>[-\d.]+)\s+GY:(?P<gy>[-\d.]+)\s+GZ:(?P<gz>[-\d.]+)")

def find_port():
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    usb = [p for p in ports if "usb" in p.device.lower() or "serial" in p.device.lower()]
    return usb[0].device if usb else (ports[0].device if ports else None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture", required=True, help="up, down, left, right")
    args = parser.parse_args()
    
    gesture = args.gesture.lower()
    save_dir = os.path.join(DATA_DIR, gesture.upper())
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    else:
        # Clear existing data for a fresh start
        print(f"Purging old {gesture.upper()} data...")
        for f in os.listdir(save_dir):
            if f.endswith(".txt"): os.remove(os.path.join(save_dir, f))
    
    port = find_port()
    print(f"--- FAST COLLECTOR: {gesture.upper()} ---")
    print(f"Goal: {SAMPLES_NEEDED} samples. Trigger threshold: {TRIGGER_G}g")
    
    count = 0
    try:
        with serial.Serial(port, BAUD_RATE, timeout=0.01) as ser:
            while count < SAMPLES_NEEDED:
                print(f"\n[Ready] Perform gesture {count+1}/{SAMPLES_NEEDED}...")
                
                # Wait for trigger
                recording = []
                triggered = False
                while not triggered:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    m = LINE_RE.search(line)
                    if m:
                        ax, az = float(m.group("ax")), float(m.group("az"))
                        # Use a very generic trigger for collection
                        if abs(ax) > TRIGGER_G or abs(az) > 1.2 or abs(az) < 0.8: # Simple check
                            triggered = True
                            print("[*] Recording...", end='', flush=True)
                            recording.append(line)
                
                # Capture window
                start_t = time.time()
                while time.time() - start_t < WINDOW_SEC:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line: recording.append(line)
                
                # Save
                filename = os.path.join(save_dir, f"{gesture}_{int(time.time())}.txt")
                with open(filename, "w") as f:
                    f.write("AX,AY,AZ,GX,GY,GZ\n") # Header for pandas
                    for r in recording:
                        m = LINE_RE.search(r)
                        if m:
                            f.write(f"{m.group('ax')},{m.group('ay')},{m.group('az')},{m.group('gx')},{m.group('gy')},{m.group('gz')}\n")
                
                print(" Saved.")
                count += 1
                time.sleep(1.0) # Settle time
                
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
