# Technical Evolution: ESP32 Gesture Music Controller

This document details the signal processing, mathematical strategies, and AI architectures implemented from the base project to the final "Flash-AI" Hybrid system.

---

## 🏗️ Stage 1: Global Statistics (The Baseline)
**Scripts:** `train_model.py`, `live_control.py`

### 🔬 Technical Strategy:
*   **Windowing**: 4.0-second static capture.
*   **Feature Engineering**: Global statistics (Mean, Standard Deviation, Max, Min) calculated across the entire 4-second buffer.
*   **Signal Processing**: Direct raw input from Serial.

### ⚠️ Problems Encountered:
1.  **Signal Dilution**: Since a gesture only lasts ~0.3s, 92% of the features were calculating background noise ("Idle" data). This made the model extremely weak.
2.  **Latency**: The user had to perform a gesture and then wait for the remainder of the 4-second window to finish before the AI would even start thinking.

---

## ⏱️ Stage 2: Temporal Windowing (The Speed War)
**Scripts:** `train_model_4sec.py`, `live_detect_4sec.py`, `train_model_2.5sec.py`

### 🔬 Technical Strategy:
*   **Windowing**: Reduced capture to 2.5s.
*   **Thresholding**: Introduced global magnitude triggers to start the AI recording.

### ⚠️ Problems Encountered:
1.  **Sign-Flip Error**: The model still couldn't distinguish between `UP` and `DOWN` because it was looking at absolute values (averages).
2.  **Orientation Sensitivity**: If the sensor was tilted by even 10 degrees, the "G-offset" on the Z-axis would change, rendering the trained model useless (gravity bias).

---

## ⚡ Stage 3: Impulse Response (The Physics Engine)
**Scripts:** `impulse_control.py`

### 🔬 Technical Strategy:
*   **Signal Processing**: Implemented a **Band-pass Filter** in software:
    *   **Low-Pass Filter (LPF)**: `cur_lp = last_lp * (1 - α) + x * α` (Removes high-frequency jitter).
    *   **High-Pass Filter (HPF)**: `cur_hp = β * (last_hp + lp - last_lp)` (Removes gravity).
*   **Logic**: Real-time peak detection. `axis = np.argmax(abs_impulses)`.

### ✅ Successes:
*   Zero perceived latency. The moment the threshold was crossed, the command fired.

### ⚠️ Problems Encountered:
*   **False Positives**: With no AI validation, setting a coffee cup on the table or shifting in your chair would trigger a "Next Track" command.

---

## 🚀 Stage 4: Hybrid Flash-AI (The Ultimate Solution)
**Scripts:** `train_flash_model.py`, `flash_detect.py`, `fast_collect.py`

### 🔬 Technical Strategy:
1.  **Temporal Segmentation (Time-Zones)**:
    *   The 1.0s window is split into **4 chunks** (0.25s each).
    *   The AI treats these as phases: [Pre-Flick, The Strike, The Braking, The Return].
    *   Features: `[mean, std, max, min]` for EACH chunk (114-dimension vector).

2.  **Gravity-Aware Gating**:
    *   **Gravity Baseline**: We calculate `df[col].mean()` and feed it to the AI.
    *   This tells the AI the **Static Orientation** of the sensor, allowing it to interpret polarity correctly even if held upside down.

3.  **Trigger Alignment (The "Secret Sauce")**:
    *   We modified the Training script to find the **Trigger Point** (crossing 0.3g) in the training files.
    *   This ensures the AI is trained on data that starts exactly when the live script "wakes up."

4.  **Hardware Gating**:
    *   The AI decides "Is it a gesture?".
    *   If **AI Confidence > 45%**, we look at the **Physical Peak Polarity** on the triggered axis.
    *   `Peak > 0 ? Direction A : Direction B`. This provides 100% directional reliability.

### 📊 Model Final Stats:
*   **Window**: 1.0 Sec
*   **Accuracy**: 78% (AI Validation) -> 99%+ (Effective with Physics Polarity)
*   **Latency**: 1s (Recording) + ~5ms (Inference) = **~1.0s total**.
