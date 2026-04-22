import os
import pandas as pd
import numpy as np
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
FS = 50.0  
CUTOFF = 5.0 # We only care about movements under 5 Hz (human speed)
ORDER = 4    # Sharpness of the filter

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Applies a low-pass filter to remove high-frequency sensor noise."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply filter forwards and backwards for zero phase shift
    y = filtfilt(b, a, data)
    return y

def extract_features(filepath):
    """Reads a gesture file, filters noise, and extracts statistical features."""
    try:
        df = pd.read_csv(filepath, sep='\s+')
        
        if len(df) < 10: # Increased minimum rows slightly to ensure the filter works properly
            print(f"[WARNING] Skipping {os.path.basename(filepath)}: File is empty or too short.")
            return None

        features = []
        
        # We extract features for both Accelerometer and Gyroscope
        for col in ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']:
            raw_data = df[col].values
            
            # --- 1. APPLY FILTER ---
            # Remove jitter/noise before doing any math
            clean_data = butter_lowpass_filter(raw_data, CUTOFF, FS, ORDER)
            
            # --- 2. EXTRACT FEATURES ---
            features.extend([
                np.mean(clean_data), 
                np.std(clean_data), 
                np.max(clean_data), 
                np.min(clean_data)
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