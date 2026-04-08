import time
import serial
import serial.tools.list_ports
import re
from pynput.keyboard import Key, Controller
import numpy as np

# --- CONFIGURATION ---
BAUD_RATE = 115200
FLICK_THRESHOLD = 0.8  # G-force required to trigger
COOLDOWN = 0.5         # Seconds to wait between triggers
ALPHA = 0.2            # Low-pass filter for gravity removal

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
    port = find_port()
    if not port:
        print("No serial port found."); return

    # Simple state for gravity removal (DC offset)
    gravity = np.array([0.0, 0.0, 0.0])
    last_trigger = 0

    print(f"--- INSTANT ENERGY CONTROL STARTING (Thresh: {FLICK_THRESHOLD}g) ---")
    print("Flick Directions:")
    print("  RIGHT/LEFT (X-Axis) -> Next/Prev")
    print("  UP/DOWN    (Z-Axis) -> Vol Up/Down")
    
    try:
        with serial.Serial(port, BAUD_RATE, timeout=0.1) as ser:
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line: continue
                
                m = LINE_RE.search(line)
                if m:
                    accel = np.array([float(m.group("ax")), float(m.group("ay")), float(m.group("az"))])
                    
                    # 1. Update Gravity estimate (Slow moving average)
                    gravity = gravity * (1-ALPHA) + accel * ALPHA
                    
                    # 2. Get "Energy" (Accleration without gravity)
                    energy = accel - gravity
                    
                    # 3. Check for Flicks (if not in cooldown)
                    if time.time() - last_trigger > COOLDOWN:
                        # Priority Axis Check
                        ax, ay, az = energy
                        
                        # Lateral Flick (X)
                        if abs(ax) > FLICK_THRESHOLD and abs(ax) > abs(az):
                            if ax > 0:
                                print(f">>> FLICK RIGHT ({ax:.2f}g) -> Next Track")
                                keyboard.tap(Key.media_next)
                            else:
                                print(f"<<< FLICK LEFT ({ax:.2f}g) -> Prev Track")
                                keyboard.tap(Key.media_previous)
                            last_trigger = time.time()
                            
                        # Vertical Flick (Z)
                        elif abs(az) > FLICK_THRESHOLD and abs(az) > abs(ax):
                            if az > 0:
                                print(f"^^^ FLICK UP ({az:.2f}g) -> Vol Up")
                                keyboard.tap(Key.media_volume_up)
                            else:
                                print(f"vvv FLICK DOWN ({az:.2f}g) -> Vol Down")
                                keyboard.tap(Key.media_volume_down)
                            last_trigger = time.time()
                            
                time.sleep(0.001) # High frequency polling
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
