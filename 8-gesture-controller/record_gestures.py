import os
import re
import time
import sys
import threading
import argparse
import serial
import serial.tools.list_ports
import pandas as pd

# --- CONFIGURATION ---
BAUD_RATE = 115200
DEFAULT_DURATION = 2.5       # 2.5 second gesture windows
DEFAULT_SAMPLES_TARGET = 25  # Target: at least 25 samples per gesture

GESTURES = [
    'clap',
    'down',
    'idle',
    'left',
    'pull',
    'push',
    'right',
    'up',
    'wrist_rotate_left',
    'wrist_rotate_right'
]

# ANSI Colors for a premium terminal UI
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
                # Flush initial garbage
                ser.flushInput()
                t0 = time.perf_counter()
                while self.running:
                    raw = ser.readline()
                    decoded = raw.decode("utf-8", errors="replace").strip()
                    parsed = parse_line(decoded)
                    if parsed:
                        now = time.perf_counter() - t0
                        with self.lock:
                            self.t.append(now)
                            self.ax.append(parsed[0]); self.ay.append(parsed[1]); self.az.append(parsed[2])
                            self.gx.append(parsed[3]); self.gy.append(parsed[4]); self.gz.append(parsed[5])
        except Exception as e:
            print(f"\n{C_RED}Serial Error: {e}{C_RESET}")

def get_existing_count(directory, gesture):
    """Counts files matching gesture_{index}.txt in directory."""
    if not os.path.exists(directory):
        return 0
    pattern = re.compile(rf"^{re.escape(gesture)}_\d+\.txt$")
    files = [f for f in os.listdir(directory) if pattern.match(f)]
    return len(files)

def print_status_table(data_dir):
    """Prints a beautiful summary table of current counts across all gestures."""
    print(f"\n{C_BOLD}{C_CYAN}Current Collection Status at: {data_dir}{C_RESET}")
    print(f"{C_CYAN}+----------------------+---------------+---------+{C_RESET}")
    print(f"{C_CYAN}| Gesture Name         | Samples Count | Target  |{C_RESET}")
    print(f"{C_CYAN}+----------------------+---------------+---------+{C_RESET}")
    for g in GESTURES:
        g_dir = os.path.join(data_dir, g)
        count = get_existing_count(g_dir, g)
        status_color = C_GREEN if count >= DEFAULT_SAMPLES_TARGET else C_YELLOW
        target_str = f"{DEFAULT_SAMPLES_TARGET}"
        print(f"| {g:20} | {status_color}{count:13d}{C_CYAN} | {target_str:7} |")
    print(f"{C_CYAN}+----------------------+---------------+---------+{C_RESET}\n")

def main():
    parser = argparse.ArgumentParser(description="Interactive 2.5s Gesture Data Recorder for 8-Gesture-Controller")
    parser.add_argument("--data-dir", default=None, help="Custom data directory (defaults to 8-gesture-controller/data)")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help=f"Duration of window in seconds (default {DEFAULT_DURATION}s)")
    args = parser.parse_args()

    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
    else:
        # Defaults to 'data' directory in '8-gesture-controller'
        data_dir = os.path.join(script_dir, "data")
    
    os.makedirs(data_dir, exist_ok=True)

    # Automatically find serial port
    port = find_port()
    print(f"\n{C_BOLD}{C_CYAN}=== 8-GESTURE-CONTROLLER RECORDER ==={C_RESET}")
    print(f"Configured Duration : {C_BOLD}{args.duration} seconds{C_RESET}")
    print(f"Output Directory    : {C_BOLD}{data_dir}{C_RESET}")
    
    if not port:
        print(f"\n{C_RED}{C_BOLD}Error: No serial device found.{C_RESET}")
        print("Please ensure your ESP32/IMU is connected and printing data.")
        sys.exit(1)
    print(f"Serial Port Detected: {C_GREEN}{port}{C_RESET}\n")

    while True:
        # Show summary status
        print_status_table(data_dir)
        
        print(f"{C_BOLD}Select a Gesture to Record:{C_RESET}")
        for idx, g in enumerate(GESTURES, 1):
            g_dir = os.path.join(data_dir, g)
            count = get_existing_count(g_dir, g)
            status_tag = f"{C_GREEN}[DONE]{C_RESET}" if count >= DEFAULT_SAMPLES_TARGET else f"{C_YELLOW}[COLLECTING]{C_RESET}"
            print(f"  {C_CYAN}{idx:2d}){C_RESET} {g:22} {status_tag} ({count}/{DEFAULT_SAMPLES_TARGET} samples)")
        print(f"  {C_RED}q) Exit{C_RESET}")

        choice = input(f"\nEnter choice (1-{len(GESTURES)} or q): ").strip()
        if choice.lower() == 'q':
            print(f"\n{C_CYAN}Exiting recorder. Happy training!{C_RESET}\n")
            break

        if not choice.isdigit() or not (1 <= int(choice) <= len(GESTURES)):
            print(f"\n{C_RED}Invalid choice. Please select a valid number or 'q'.{C_RESET}")
            time.sleep(1.5)
            continue

        selected_gesture = GESTURES[int(choice) - 1]
        gesture_dir = os.path.join(data_dir, selected_gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        print(f"\n{C_BOLD}{C_CYAN}--- Recording Menu for '{selected_gesture.upper()}' ---{C_RESET}")
        
        while True:
            current_count = get_existing_count(gesture_dir, selected_gesture)
            print(f"\nCurrent samples for {C_BOLD}{selected_gesture}{C_RESET}: {C_GREEN if current_count >= DEFAULT_SAMPLES_TARGET else C_YELLOW}{current_count}{C_RESET}")
            
            action = input(f"Press {C_BOLD}ENTER{C_RESET} to record sample #{current_count}, type {C_BOLD}'b'{C_RESET} to go back: ").strip()
            if action.lower() == 'b':
                break

            filename = f"{selected_gesture}_{current_count:02d}.txt"
            filepath = os.path.join(gesture_dir, filename)

            # Countdown
            print(f"\n{C_YELLOW}Get ready!{C_RESET}")
            for count in [3, 2, 1]:
                print(f"{C_BOLD}{count}...{C_RESET}")
                time.sleep(1)
            
            print(f"\n{C_GREEN}{C_BOLD}=== RECORDING NOW ({args.duration}s) ==={C_RESET}")
            
            # Start serial reader
            reader = SerialReader(port, BAUD_RATE)
            reader.start()
            
            # Visual progress bar during recording
            steps = 10
            step_duration = args.duration / steps
            for s in range(steps):
                progress = "=" * (s + 1) + " " * (steps - s - 1)
                print(f"\r[{C_CYAN}{progress}{C_RESET}] {int((s+1)/steps*100)}%", end='', flush=True)
                time.sleep(step_duration)
            
            print("\n")
            
            # Stop serial reader
            reader.running = False
            reader.join()

            with reader.lock:
                if len(reader.t) < 10:
                    print(f"{C_RED}{C_BOLD}Error: No IMU data captured.{C_RESET}")
                    print("Check your ESP32 serial output and ensure BAUD rate is matching.")
                    continue
                
                df = pd.DataFrame({
                    'Time': reader.t,
                    'AX': reader.ax, 'AY': reader.ay, 'AZ': reader.az,
                    'GX': reader.gx, 'GY': reader.gy, 'GZ': reader.gz
                })

            # Save the recorded data
            df.to_csv(filepath, index=False, sep=' ')
            print(f"{C_GREEN}Saved {C_BOLD}{filename}{C_RESET} successfully! ({len(df)} samples captured)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{C_RED}Session interrupted. Exiting safely...{C_RESET}\n")
        sys.exit(0)
