import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model") # New model directory

def extract_features(filepath):
    """Reads a gesture file and extracts statistical features."""
    try:
        df = pd.read_csv(filepath, sep='\s+')
        
        # --- NEW SAFEGUARD 1: Check if file has enough data ---
        # We need at least 2 rows to calculate standard deviation
        if len(df) < 2:
            print(f"[WARNING] Skipping {os.path.basename(filepath)}: File is empty or too short.")
            return None

        features = []
        for col in ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']:
            features.extend([
                df[col].mean(), 
                df[col].std(), 
                df[col].max(), 
                df[col].min()
            ])
            
        # --- NEW SAFEGUARD 2: Check for any weird math errors (NaNs) ---
        if np.isnan(features).any():
            print(f"[WARNING] Skipping {os.path.basename(filepath)}: Contains invalid NaN math data.")
            return None
            
        return features
        
    except Exception as e:
        print(f"Error reading {os.path.basename(filepath)}: {e}")
        return None

def main():
    # Ensure the model directory exists before we try to save anything
    os.makedirs(MODEL_DIR, exist_ok=True)

    X = []
    y = []
    
    print("Loading data and extracting features...")
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
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining SVM model...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))
    
    # Save Model and Scaler into the new model folder
    model_path = os.path.join(MODEL_DIR, 'gesture_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'gesture_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nSaved 'gesture_model.pkl' and 'gesture_scaler.pkl' inside the '{MODEL_DIR}' folder.")

if __name__ == "__main__":
    main()