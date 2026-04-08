import math
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data_flash")
DATA_DIR_ALT = os.path.join(SCRIPT_DIR, "data_4sec")
IDLE_DIR = os.path.join(SCRIPT_DIR, "data", "idle")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

FS = 100.0  
WINDOW_DURATION = 1.0 # Sharp 1s burst
LPF_ALPHA = 0.3
HPF_ALPHA = 0.95

def extract_features(filepath):
    try:
        df = pd.read_csv(filepath, sep=None, engine="python")
        
        # --- FILTERS & PEAK FINDING ---
        ax_dt = df["AX"].values - df["AX"].mean()
        ay_dt = df["AY"].values - df["AY"].mean()
        az_dt = df["AZ"].values - df["AZ"].mean()
        mags = np.sqrt(ax_dt**2 + ay_dt**2 + az_dt**2)
        
        trigger_indices = np.where(mags > 0.3)[0]
        start = max(0, trigger_indices[0] - int(FS * 0.1)) if len(trigger_indices) > 0 else 0
        df = df.iloc[start : start + int(FS * WINDOW_DURATION)]
        
        # Padding
        if len(df) < int(FS * WINDOW_DURATION):
            pad = int(FS * WINDOW_DURATION) - len(df)
            df = pd.concat([df, pd.concat([df.iloc[-1:]]*pad)], ignore_index=True)

        channels = ["AX", "AY", "AZ", "GX", "GY", "GZ"]
        features = []
        
        for col in channels:
            data = df[col].values
            
            # Band-pass Filter
            lp = data[0]
            lpf = []
            for x in data:
                lp = lp * (1-LPF_ALPHA) + x * LPF_ALPHA
                lpf.append(lp)
            
            hp = 0
            hpf = []
            last_lp_val = lpf[0]
            for lp_val in lpf:
                hp = HPF_ALPHA * (hp + lp_val - last_lp_val)
                hpf.append(hp)
                last_lp_val = lp_val
            
            chan_data = np.array(hpf)
            
            # Capture SIGNED pattern (No Unit Scaling)
            chunks = np.array_split(chan_data, 4)
            for chunk in chunks:
                if len(chunk) == 0:
                    features.extend([0, 0, 0, 0])
                else:
                    features.extend([np.mean(chunk), np.std(chunk), np.max(chunk), np.min(chunk)])
            
            # Orientation and Intensity
            features.extend([np.sum(chan_data), np.max(np.abs(chan_data)), df[col].mean()])
        
        return features
    except Exception as e:
        return None

def main():
    X, y = [], []
    print(f"Loading data for Flash-AI...")
    
    for base_dir in [DATA_DIR, DATA_DIR_ALT]:
        if not os.path.exists(base_dir): continue
        for gesture in ["up", "down", "left", "right"]:
            gp = os.path.join(base_dir, gesture.upper())
            if not os.path.exists(gp): continue
            for filename in os.listdir(gp):
                if filename.endswith(".txt"):
                    feat = extract_features(os.path.join(gp, filename))
                    if feat: X.append(feat); y.append(gesture)
                
    if os.path.exists(IDLE_DIR):
        for f in os.listdir(IDLE_DIR):
            if f.endswith(".txt"):
                feat = extract_features(os.path.join(IDLE_DIR, f))
                if feat: X.append(feat); y.append("idle")

    # Training
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.15, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Final Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
    
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    joblib.dump(model, os.path.join(MODEL_DIR, "flash_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "flash_scaler.pkl"))

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Flash Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
    
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    joblib.dump(model, os.path.join(MODEL_DIR, "flash_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "flash_scaler.pkl"))

if __name__ == "__main__":
    main()
