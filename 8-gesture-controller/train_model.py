import math
import pandas as pd
import numpy as np
import os
import argparse
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
FS = 100.0  
WINDOW_DURATION = 2.5 # 2.5 second gesture windows
CUTOFF = 5.0
ORDER = 4
MADGWICK_BETA = 0.1

# ANSI Colors
C_GREEN = "\033[92m"
C_CYAN = "\033[96m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)

class Madgwick:
    def __init__(self, sample_freq=100.0, beta=0.1):
        self.sample_freq = sample_freq
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, accel, gyro):
        gx, gy, gz = gyro
        ax, ay, az = accel
        q = self.q
        norm = np.linalg.norm(accel)
        if norm == 0: return
        ax /= norm; ay /= norm; az /= norm
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
        if step_norm > 0: step /= step_norm
        else: step = np.zeros(4)
        q_dot = 0.5 * self.q_mult(q, [0, gx, gy, gz]) - self.beta * step
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

def extract_features(filepath):
    try:
        # Auto-detect comma or space separator
        df = pd.read_csv(filepath, sep=None, engine="python")
        target_len = int(FS * WINDOW_DURATION)
        
        # Make sure columns are uppercase
        df.columns = [c.upper() for c in df.columns]
        
        # Check if required columns are present
        required_cols = ["AX", "AY", "AZ", "GX", "GY", "GZ"]
        if not all(col in df.columns for col in required_cols):
            # Try to handle case where columns might be lowercase or parsed incorrectly
            col_mapping = {c: c.upper() for c in df.columns}
            df = df.rename(columns=col_mapping)
            if not all(col in df.columns for col in required_cols):
                return None

        if len(df) < target_len:
            padding_len = target_len - len(df)
            last_rows = df.iloc[-1:].iloc[np.zeros(padding_len)].copy()
            df = pd.concat([df, last_rows], ignore_index=True)
        else:
            df = df.iloc[:target_len]

        processed_channels = {}
        for col in required_cols:
            processed_channels[col] = butter_lowpass_filter(df[col].values, CUTOFF, FS, ORDER)
        
        fusion = Madgwick(sample_freq=FS, beta=MADGWICK_BETA)
        orientation_history = []
        for i in range(len(df)):
            accel = [processed_channels["AX"][i], processed_channels["AY"][i], processed_channels["AZ"][i]]
            gyro_rad = [processed_channels[c][i] * (math.pi/180.0) for c in ["GX", "GY", "GZ"]]
            fusion.update(accel, gyro_rad)
            orientation_history.append(fusion.get_euler())
            
        orientation_history = np.array(orientation_history)
        all_channels = [
            processed_channels["AX"], processed_channels["AY"], processed_channels["AZ"],
            processed_channels["GX"], processed_channels["GY"], processed_channels["GZ"],
            orientation_history[:, 0], orientation_history[:, 1], orientation_history[:, 2]
        ]
        
        features = []
        for data in all_channels:
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val > 0.0001:
                skew = np.mean((data - mean_val) ** 3) / (std_val ** 3)
                kurt = np.mean((data - mean_val) ** 4) / (std_val ** 4) - 3.0
            else:
                skew = 0.0
                kurt = 0.0

            half_len = len(data) // 2
            half1 = data[:half_len]
            half2 = data[half_len:]

            features.extend([
                mean_val,
                std_val,
                np.max(data),
                np.min(data),
                np.sum(data),
                np.sum(data**2),
                np.max(data) - np.min(data),
                np.sum(np.abs(data)),
                np.sum(np.abs(np.diff(data))),
                skew,
                kurt,
                float(np.mean(half1) - np.mean(half2))
            ])
        return features
    except Exception as e:
        print(f"{C_RED}Error processing {filepath}: {e}{C_RESET}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Train model on 2.5s gesture data")
    parser.add_argument("--data-dirs", nargs='+', default=None, help="Custom data directories (space separated, defaults to 8-gesture-controller/data)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of files to process per gesture per directory (default: load all files)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_dirs = []
    if args.data_dirs:
        data_dirs = [os.path.abspath(d) for d in args.data_dirs]
    else:
        data_dirs = [os.path.join(script_dir, "data")]

    print(f"\n{C_BOLD}{C_CYAN}=== TRAINING 8-GESTURE MODEL ==={C_RESET}")
    print(f"Data Directories: {C_BOLD}{', '.join(data_dirs)}{C_RESET}")

    X, y = [], []
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"{C_RED}Warning: Data directory '{data_dir}' does not exist. Skipping.{C_RESET}")
            continue

        gestures_found = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

        if not gestures_found:
            print(f"{C_RED}Warning: No gesture subdirectories found in '{data_dir}'. Skipping.{C_RESET}")
            continue

        print(f"\nLoading gesture files from {data_dir}...")
        for gesture in gestures_found:
            gesture_path = os.path.join(data_dir, gesture)
            files = [f for f in os.listdir(gesture_path) if f.endswith(".txt")]
            # Sort numerically by extracting the digit from 'gesture_X.txt' to keep the selection stable
            import re
            files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
            if "friends_data" in data_dir:
                # Keep all files
                pass
            else:
                # Only use the first 30 files of user personal data
                files = files[:30]
            print(f"  * {gesture:20} : Processing {len(files)} files", end='', flush=True)
            
            success_count = 0
            for filename in files:
                filepath = os.path.join(gesture_path, filename)
                feat = extract_features(filepath)
                if feat:
                    X.append(feat)
                    y.append(gesture)
                    success_count += 1
            print(f" -> Processed {success_count} successfully.")

    if not X:
        print(f"{C_RED}Error: No training data processed successfully.{C_RESET}")
        return

    print(f"\n{C_GREEN}Successfully processed {len(X)} total gesture samples.{C_RESET}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    print(f"\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=200, max_depth=16, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{C_GREEN}{C_BOLD}Training Completed Successfully!{C_RESET}")
    print(f"{C_CYAN}Overall Test Accuracy: {C_BOLD}{acc * 100:.2f}%{C_RESET}")

    print(f"\n{C_BOLD}Classification Report:{C_RESET}")
    print(classification_report(y_test, y_pred))

    print(f"\n{C_BOLD}Confusion Matrix:{C_RESET}")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    print(cm_df.to_string())

    # Save Model, Scaler, and Confusion Matrix Image
    model_dir = os.path.join(script_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    cm_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Gesture Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    model_path = os.path.join(model_dir, "gesture_model_2.5sec.pkl")
    scaler_path = os.path.join(model_dir, "gesture_scaler_2.5sec.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n{C_GREEN}Model saved as: {C_BOLD}{model_path}{C_RESET}")
    print(f"{C_GREEN}Scaler saved as: {C_BOLD}{scaler_path}{C_RESET}")
    print(f"{C_GREEN}Confusion matrix saved as: {C_BOLD}{cm_path}{C_RESET}\n")

if __name__ == "__main__":
    main()
