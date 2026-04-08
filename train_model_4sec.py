import math
import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_4SEC = os.path.join(SCRIPT_DIR, "data_4sec")
DATA_IDLE = os.path.join(SCRIPT_DIR, "data", "idle")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

FS = 100.0  
WINDOW_DURATION = 4.0 
CUTOFF = 5.0
ORDER = 4
MADGWICK_BETA = 0.1

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
        df = pd.read_csv(filepath, sep=None, engine="python")
        target_len = int(FS * WINDOW_DURATION)
        
        if len(df) < target_len:
            padding_len = target_len - len(df)
            last_rows = df.iloc[-1:].iloc[np.zeros(padding_len)].copy()
            df = pd.concat([df, last_rows], ignore_index=True)
        else:
            df = df.iloc[:target_len]

        processed_channels = {}
        for col in ["AX", "AY", "AZ", "GX", "GY", "GZ"]:
            raw_data = df[col].values
            # Filter then Detrend (subtract mean to remove gravity bias)
            filtered = butter_lowpass_filter(raw_data, CUTOFF, FS, ORDER)
            processed_channels[col] = filtered - np.mean(filtered)
        
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
            # Stats (Mean, Std, Max, Min)
            features.extend([np.mean(data), np.std(data), np.max(data), np.min(data)])
            # Physics (Net Impulse, Peak Energy)
            features.append(np.sum(data))       # Impulse
            features.append(np.sum(data**2))    # Total Energy
            features.append(np.max(np.abs(data))) # Peak Force
        return features
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    X, y = [], []
    print(f"Loading data from {DATA_4SEC}...")
    for gesture in os.listdir(DATA_4SEC):
        gesture_path = os.path.join(DATA_4SEC, gesture)
        if not os.path.isdir(gesture_path): continue
        for filename in os.listdir(gesture_path):
            if filename.endswith(".txt"):
                feat = extract_features(os.path.join(gesture_path, filename))
                if feat:
                    X.append(feat); y.append(gesture.lower())
                    
    print(f"Loading idle data from {DATA_IDLE}...")
    if os.path.exists(DATA_IDLE):
        for filename in os.listdir(DATA_IDLE):
            if filename.endswith(".txt"):
                feat = extract_features(os.path.join(DATA_IDLE, filename))
                if feat:
                    X.append(feat); y.append("idle")

    if not X:
        print("No data found!"); return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
    
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    joblib.dump(model, os.path.join(MODEL_DIR, "gesture_model_4sec.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "gesture_scaler_4sec.pkl"))
    print("Model saved as model/gesture_model_4sec.pkl")

if __name__ == "__main__":
    main()
