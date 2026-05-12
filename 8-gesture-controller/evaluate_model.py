"""
Evaluate the saved gesture model against the training data.

Loads the saved model + scaler, re-extracts features from the data directories,
reproduces the same 80/20 split, and prints full evaluation metrics.

Usage:
    python evaluate_model.py
    python evaluate_model.py --data-dirs data friends_data
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from train_model import extract_features

# ANSI Colors
C_GREEN = "\033[92m"
C_CYAN = "\033[96m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved gesture model")
    parser.add_argument("--data-dirs", nargs='+', default=None,
                        help="Data directories (defaults to 'data')")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model", "gesture_model_2.5sec.pkl")
    scaler_path = os.path.join(script_dir, "model", "gesture_scaler_2.5sec.pkl")

    # ── Load model & scaler ──
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"{C_RED}Error: Model or scaler not found in model/ directory.{C_RESET}")
        print("Run train_model.py first.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"{C_GREEN}✓ Model loaded:{C_RESET}  {model_path}")
    print(f"{C_GREEN}✓ Scaler loaded:{C_RESET} {scaler_path}")

    # ── Model info ──
    print(f"\n{C_BOLD}{C_CYAN}══ MODEL INFO ══{C_RESET}")
    print(f"  Type             : {type(model).__name__}")
    print(f"  Num estimators   : {model.n_estimators}")
    print(f"  Classes          : {list(model.classes_)}")
    print(f"  Num features     : {model.n_features_in_}")

    # ── Load data ──
    data_dirs = []
    if args.data_dirs:
        data_dirs = [os.path.abspath(d) for d in args.data_dirs]
    else:
        data_dirs = [os.path.join(script_dir, "data")]

    X, y = [], []
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"{C_YELLOW}Warning: '{data_dir}' not found, skipping.{C_RESET}")
            continue

        gestures = sorted([d for d in os.listdir(data_dir)
                           if os.path.isdir(os.path.join(data_dir, d))])
        print(f"\n{C_CYAN}Loading from {data_dir}:{C_RESET}")
        for gesture in gestures:
            gesture_path = os.path.join(data_dir, gesture)
            files = sorted([f for f in os.listdir(gesture_path) if f.endswith(".txt")])
            files = files[:30]  # same cap as train_model.py
            success = 0
            for f in files:
                feat = extract_features(os.path.join(gesture_path, f))
                if feat:
                    X.append(feat)
                    y.append(gesture)
                    success += 1
            print(f"  {gesture:22} : {success}/{len(files)} samples extracted")

    if not X:
        print(f"{C_RED}No data found.{C_RESET}")
        return

    X = np.array(X)
    y = np.array(y)
    print(f"\n{C_GREEN}Total samples: {len(X)}{C_RESET}")

    # ── Reproduce the same 80/20 split (random_state=42) ──
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ── Evaluate on test set ──
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{C_BOLD}{C_CYAN}══ TEST SET RESULTS (80/20 split, random_state=42) ══{C_RESET}")
    print(f"  Train samples : {len(X_train)}")
    print(f"  Test samples  : {len(X_test)}")
    print(f"\n  {C_BOLD}Overall Accuracy: {C_GREEN}{acc * 100:.2f}%{C_RESET}")

    print(f"\n{C_BOLD}Classification Report:{C_RESET}")
    print(classification_report(y_test, y_pred, zero_division=0))

    print(f"{C_BOLD}Confusion Matrix:{C_RESET}")
    labels = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())

    # ── Full dataset accuracy (train + test) ──
    y_pred_all = model.predict(X_scaled)
    acc_all = accuracy_score(y, y_pred_all)
    print(f"\n{C_BOLD}Full Dataset Accuracy (all {len(X)} samples): {C_GREEN}{acc_all * 100:.2f}%{C_RESET}")

    # ── 5-fold cross-validation ──
    print(f"\n{C_BOLD}{C_CYAN}══ 5-FOLD CROSS-VALIDATION ══{C_RESET}")
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"  Fold accuracies : {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"  {C_BOLD}Mean CV Accuracy : {C_GREEN}{cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%{C_RESET}")

    # ── Per-class summary ──
    print(f"\n{C_BOLD}{C_CYAN}══ PER-CLASS SUMMARY ══{C_RESET}")
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )
    print(f"  {'Gesture':22} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Support':>8}")
    print(f"  {'-'*54}")
    for i, label in enumerate(labels):
        print(f"  {label:22} {prec[i]*100:6.1f}% {rec[i]*100:6.1f}% {f1[i]*100:6.1f}% {support[i]:7d}")


if __name__ == "__main__":
    main()
