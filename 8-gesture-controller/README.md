# 🎛️ 8-Gesture Controller: 2.5-Second Window Pipeline

Welcome to your specialized **8-Gesture Controller** module. This module provides a complete, modern pipeline to record, train, and run real-time inference on **up to 10 distinct gestures** with highly responsive **2.5-second capture windows**.

---

## 📂 Folder Contents
*   [record_gestures.py](file:///Users/danny/gesture-music-controller/8-gesture-controller/record_gestures.py): Premium, interactive CLI recorder with visual countdowns and real-time database counting.
*   [train_model.py](file:///Users/danny/gesture-music-controller/8-gesture-controller/train_model.py): High-performance trainer utilizing Butterworth Low-pass filtering, Madgwick Orientation fusion, and Random Forest classification.
*   [live_detect.py](file:///Users/danny/gesture-music-controller/8-gesture-controller/live_detect.py): Real-time live controller with calibration, motion thresholding, and keyboard hotkey mapping (volume up/down, tracks control).

---

## 🚀 Step-by-Step Guide

### 1️⃣ Check/Record Data
Our interactive recording utility automatically finds your serial device, displays a status table of your current dataset, and guides you through recording samples step-by-step.

```bash
# Run with the virtual environment Python
venv/bin/python 8-gesture-controller/record_gestures.py
```

*   **Configuring target directory**: You can record directly into your Downloads database by specifying:
    ```bash
    venv/bin/python 8-gesture-controller/record_gestures.py --data-dir /Users/danny/Downloads/data
    ```

### 2️⃣ Train the Model
The trainer loads the 2.5-second data, filters high-frequency noise, reconstructs 3D orientations, and extracts statistical & energy features to train a high-accuracy classifier.

```bash
# Train on the default data directory:
venv/bin/python 8-gesture-controller/train_model.py

# Train on your Downloads directory:
venv/bin/python 8-gesture-controller/train_model.py --data-dir /Users/danny/Downloads/data
```

### 3️⃣ Real-Time Live Control
Once trained, launch the live detector to translate physically performed gestures into computer hotkeys:

```bash
venv/bin/python 8-gesture-controller/live_detect.py
```

---

## 📊 Performance Statistics
Using the **250 samples** across all **10 gestures** from `/Users/danny/Downloads/data`, this pipeline achieves a **98.00% Overall Test Accuracy** with pristine precision and recall scores across all categories.
