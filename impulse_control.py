import time
import serial
import serial.tools.list_ports
import re
from pynput.keyboard import Key, Controller
import numpy as np

# --- CONFIGURATION ---
BAUD_RATE = 115200
TRIGGER_G = 0.30      # Higher trigger to ignore noise
IMPULSE_MIN = 1.0     # Balanced force
COOLDOWN = 1.0        
CERTAINTY_MIN = 70    # More forgiving directional requirement
HPF_ALPHA = 0.95      # High-pass alpha
LPF_ALPHA = 0.3       # Low-pass alpha (muffles jitter)

keyboard = Controller()

LINE_RE = re.compile(
    r"AX:(?P<ax>[-\d.]+)\s+AY:(?P<ay>[-\d.]+)\s+AZ:(?P<az>[-\d.]+)"
)

def find_port():
    ports = serial.tools.list_ports.comports()
    usb = [p for p in ports if "usb" in p.device.lower() or "serial" in p.device.lower()]
    return usb[0].device if usb else (ports[0].device if ports else None)

def main():
    port = find_port()
    if not port:
        print("No serial port found."); return

    # Filter state
    last_raw = np.array([0.0, 0.0, 0.0])
    last_low_passed = np.array([0.0, 0.0, 0.0])
    last_hpf = np.array([0.0, 0.0, 0.0])
    
    last_action_time = 0
    integration_buffer = []
    state = "READY"

    print(f"--- FILTERED IMPULSE ENGINE STARTING ---")
    print("Logic: Low-Pass (Jitter) -> High-Pass (Gravity) -> Pulse")
    
    try:
        with serial.Serial(port, BAUD_RATE, timeout=0.01) as ser:
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line: continue
                
                m = LINE_RE.search(line)
                if not m: continue
                
                # 1. Band-Pass Filter Strategy
                raw = np.array([float(m.group("ax")), 0.0, float(m.group("az"))])
                
                # A. Low Pass (Muffle Jitter)
                low_passed = last_low_passed * (1 - LPF_ALPHA) + raw * LPF_ALPHA
                
                # B. High Pass (Remove Gravity)
                filtered = HPF_ALPHA * (last_hpf + low_passed - last_low_passed)
                
                last_low_passed = low_passed
                last_hpf = filtered
                last_raw = raw
                
                # 2. State Machine
                mag = np.abs(filtered)
                max_axis = np.argmax(mag)
                peak_val = filtered[max_axis]
                
                if state == "READY":
                    if np.max(mag) > TRIGGER_G and (time.time() - last_action_time) > COOLDOWN:
                        state = "PULSING"
                        integration_buffer = [filtered]
                        active_axis = max_axis
                        print(f"\n[*] Trigger: Axis {active_axis} Pulse...", end='', flush=True)
                
                elif state == "PULSING":
                    # Check if sign flipped (Braking phase started)
                    if np.sign(filtered[active_axis]) != np.sign(integration_buffer[0][active_axis]):
                        # Integration finished at peak velocity
                        total_impulse = np.sum(integration_buffer, axis=0)
                        impulse = total_impulse[active_axis]
                        abs_impulse = np.abs(total_impulse)
                        
                        print(f" Impulse: {impulse:.2f}")
                        
                        if abs(impulse) > IMPULSE_MIN:
                            # Calculate Certainty
                            total_mag = np.sum(abs_impulse)
                            certainty = (abs_impulse[active_axis] / total_mag) * 100 if total_mag > 0 else 0
                            
                            if certainty >= CERTAINTY_MIN:
                                if active_axis == 0: # X
                                    if impulse > 0:
                                        print(f">>> RIGHT (Force: {impulse:.2f}, Certainty: {certainty:.0f}%)")
                                        keyboard.tap(Key.media_next)
                                    else:
                                        print(f"<<< LEFT (Force: {impulse:.2f}, Certainty: {certainty:.0f}%)")
                                        keyboard.tap(Key.media_previous)
                                elif active_axis == 2: # Z
                                    if impulse > 0: 
                                        print(f"^^^ UP (Force: {impulse:.2f}, Certainty: {certainty:.0f}%)")
                                        keyboard.tap(Key.media_volume_up)
                                    else:
                                        print(f"vvv DOWN (Force: {impulse:.2f}, Certainty: {certainty:.0f}%)")
                                        keyboard.tap(Key.media_volume_down)
                            else:
                                print(f"[REJECT] Direction too messy ({certainty:.0f}%)")
                        else:
                            print(f"[REJECT] Too weak ({abs(impulse):.2f})")
                        
                        state = "READY"
                        last_action_time = time.time()
                    else:
                        integration_buffer.append(filtered)
                        # Safety timeout or noise floor flip
                        # Only flip sign if magnitude is decent (avoids noise-flip)
                        if len(integration_buffer) > 50 or (abs(filtered[active_axis]) < 0.05 and len(integration_buffer) > 5):
                            state = "READY"
                            if len(integration_buffer) > 50: print(" [TIMEOUT]")
                
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
