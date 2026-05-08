import os
import sys
import time
import tempfile
import pandas as pd
import numpy as np
import joblib

# Import needed parts from existing scripts
from record_gestures import SerialReader, find_port, BAUD_RATE
from train_model import extract_features

C_GREEN = "\033[92m"
C_CYAN = "\033[96m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"

DURATION = 2.5

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model", "gesture_model_2.5sec.pkl")
    scaler_path = os.path.join(script_dir, "model", "gesture_scaler_2.5sec.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"{C_RED}Model or scaler not found. Please train the model first.{C_RESET}")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    port = find_port()
    if not port:
        print(f"{C_RED}No serial device found.{C_RESET}")
        return

    print(f"\n{C_BOLD}{C_CYAN}=== MANUAL GESTURE TESTER ==={C_RESET}")
    print(f"Connected to: {C_GREEN}{port}{C_RESET}")
    
    while True:
        action = input(f"\nPress {C_BOLD}ENTER{C_RESET} to test a gesture, or 'q' to quit: ").strip()
        if action.lower() == 'q':
            break

        print(f"\n{C_YELLOW}Get ready!{C_RESET}")
        for count in [3, 2, 1]:
            print(f"{C_BOLD}{count}...{C_RESET}")
            time.sleep(1)
        
        print(f"\n{C_GREEN}{C_BOLD}=== RECORDING NOW ({DURATION}s) ==={C_RESET}")
        
        reader = SerialReader(port, BAUD_RATE)
        reader.start()
        
        # Visual progress bar during recording
        steps = 10
        step_duration = DURATION / steps
        for s in range(steps):
            progress = "=" * (s + 1) + " " * (steps - s - 1)
            print(f"\r[{C_CYAN}{progress}{C_RESET}] {int((s+1)/steps*100)}%", end='', flush=True)
            time.sleep(step_duration)
        
        print("\n")
        
        reader.running = False
        reader.join()

        with reader.lock:
            if len(reader.t) < 10:
                print(f"{C_RED}{C_BOLD}Error: No IMU data captured.{C_RESET}")
                continue
            
            df = pd.DataFrame({
                'Time': reader.t,
                'AX': reader.ax, 'AY': reader.ay, 'AZ': reader.az,
                'GX': reader.gx, 'GY': reader.gy, 'GZ': reader.gz
            })

        # Save to temp file so we can reuse extract_features which expects a filepath
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_filepath = tmp.name
        
        df.to_csv(tmp_filepath, index=False, sep=' ')
        
        features = extract_features(tmp_filepath)
        os.remove(tmp_filepath)

        if not features:
            print(f"{C_RED}Failed to extract features.{C_RESET}")
            continue

        features_scaled = scaler.transform([features])
        probs = model.predict_proba(features_scaled)[0]
        max_idx = np.argmax(probs)
        prediction = model.classes_[max_idx]
        confidence = probs[max_idx]

        print(f"{C_BOLD}PREDICTION: {C_GREEN}{prediction.upper()}{C_RESET}")
        print(f"Confidence: {confidence * 100:.2f}%")
        
        print("\nTop 3 Predictions:")
        top3_indices = np.argsort(probs)[::-1][:3]
        for idx in top3_indices:
            print(f"  {model.classes_[idx]:20}: {probs[idx]*100:.2f}%")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Exiting...{C_RESET}")
