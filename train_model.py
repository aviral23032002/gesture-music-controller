import math
import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

# --- FILTER CONFIGURATION ---
# Assume a sampling rate of ~50 Hz (adjust if your ESP32 is faster/slower)
FS = 100.0  # Matches main.c 10 ms period
WINDOW_DURATION = 2.5 # Goldilocks window
CUTOFF = 5.0 # We only care about movements under 5 Hz (human speed)
ORDER = 4    # Sharpness of the filter
MADGWICK_BETA = 0.1

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Applies a low-pass filter to remove high-frequency sensor noise."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply filter forwards and backwards for zero phase shift
    y = filtfilt(b, a, data)
    return y

class Madgwick:
    """Madgwick AHRS implementation for sensor fusion."""
    def __init__(self, sample_freq=100.0, beta=0.1):
        self.sample_freq = sample_freq
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) # Quaternion [w, x, y, z]

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
        step /= np.linalg.norm(step)
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
    """Reads a gesture file, filters noise, and extracts statistical features."""
    try:
        df = pd.read_csv(filepath, sep=r'\s+')
        
        target_len = int(FS * WINDOW_DURATION) # 400 samples
        
        # Pad or trim data to match the 4s window
        if len(df) < target_len:
            # Pad with the last value (sustaining the state) or mean
            padding_len = target_len - len(df)
            last_rows = df.iloc[-1:].iloc[np.zeros(padding_len)].copy()
            df = pd.concat([df, last_rows], ignore_index=True)
        elif len(df) > target_len:
            df = df.iloc[:target_len]
            
        features = []
        
        # 1. First, apply Low-Pass Filter to all raw channels
        processed_channels = {}
        for col in ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']:
            raw_data = df[col].values
            processed_channels[col] = butter_lowpass_filter(raw_data, CUTOFF, FS, ORDER)
        
        # 2. Run Madgwick Fusion on the filtered data
        fusion = Madgwick(sample_freq=FS, beta=MADGWICK_BETA)
        orientation_history = []
        
        for i in range(len(df)):
            accel = np.array([processed_channels['AX'][i], processed_channels['AY'][i], processed_channels['AZ'][i]])
            gyro_deg = np.array([processed_channels['GX'][i], processed_channels['GY'][i], processed_channels['GZ'][i]])
            gyro_rad = gyro_deg * (math.pi / 180.0)
            
            fusion.update(accel, gyro_rad)
            orientation_history.append(fusion.get_euler())
            
        orientation_history = np.array(orientation_history) # Nx3 (Roll, Pitch, Yaw)
        
        # 3. Extract features from all 9 channels (6 raw filtered + 3 orientation)
        all_channels = [
            processed_channels['AX'], processed_channels['AY'], processed_channels['AZ'],
            processed_channels['GX'], processed_channels['GY'], processed_channels['GZ'],
            orientation_history[:, 0], orientation_history[:, 1], orientation_history[:, 2]
        ]
        
        for channel_data in all_channels:
            features.extend([
                np.mean(channel_data), 
                np.std(channel_data), 
                np.max(channel_data), 
                np.min(channel_data)
            ])
            
        if np.isnan(features).any():
            print(f"[WARNING] Skipping {os.path.basename(filepath)}: Contains invalid NaN math data.")
            return None
            
        return features
        
    except Exception as e:
        print(f"Error reading {os.path.basename(filepath)}: {e}")
        return None

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X = []
    y = []
    
    print("Loading data, applying Low-Pass Filter, and extracting features...")
    for gesture_name in os.listdir(DATA_DIR):
        gesture_dir = os.path.join(DATA_DIR, gesture_name)
        
        if not os.path.isdir(gesture_dir):
            continue
            
        for filename in os.listdir(gesture_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(gesture_dir, filename)
                features = extract_features(filepath)
                if features is not None:
                    X.append(features)
                    y.append(gesture_name)

    if not X:
        print("No valid data found! Make sure you recorded samples first.")
        return

    X = np.array(X)
    y = np.array(y)
    print(f"\nTotal healthy samples loaded: {len(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # We keep the scaler. Even though Random Forest doesn't strictly need it, 
    # it's good practice and keeps your live_control script compatible.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- UPGRADED MODEL: Random Forest ---
    print("\nTraining Random Forest model...")
    # n_estimators=100 means it creates 100 different decision trees and averages their guesses
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))
    
    # Save Model and Scaler into the model folder
    model_path = os.path.join(MODEL_DIR, 'gesture_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'gesture_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nSaved 'gesture_model.pkl' and 'gesture_scaler.pkl' inside the '{MODEL_DIR}' folder.")

if __name__ == "__main__":
    main()