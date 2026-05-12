# 🎵 Gesture Music Controller

A high-performance, real-time gesture recognition system built with an **ESP32**, an IMU sensor, and a **Python Machine Learning Backend**. Control Spotify or system audio entirely through mid-air gestures with zero perceived latency.

## 🚀 Features
- **Zero-Latency Signal Processing**: Implements software Butterworth Low-Pass and High-Pass filters to clean noisy IMU data before it reaches the AI.
- **Hybrid Flash-AI Architecture**: Combines a lightweight Machine Learning model with hardcoded physical-polarity gating. This solves "sign-flip" errors (like getting confused between "Swipe Left" and "Swipe Right") yielding **99%+ accuracy**!
- **Real-time 6-Panel Dashboard**: Built with Matplotlib, showing live waveform data, session history, confidence breakdown, and real-time Spotify metadata (track, artist, artwork, volume).
- **AppleScript Spotify Integration**: Seamlessly controls the macOS Spotify app via native AppleScript commands—no API keys required.

## 🛠️ Hardware Setup
1. **ESP32 Microcontroller**.
2. **IMU Sensor** (e.g., MPU6050/BNO085) connected via I2C or SPI.
3. ESP32 sends raw Accelerometer and Gyroscope data over Serial at 115200 baud.

## 💻 Software Setup

### 1. ESP32 Firmware
Navigate to the root directory and build the C/C++ firmware using ESP-IDF:
```bash
idf.py set-target esp32
idf.py build
idf.py flash monitor
```

### 2. Python Backend & Dashboard
The entire ML pipeline and dashboard live in the `8-gesture-controller/` directory.

**Prerequisites:**
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy scipy scikit-learn matplotlib pyserial
# Optional UI dependencies for Album Art
pip install pillow
```

**Running the System:**
```bash
python 8-gesture-controller/dashboard.py
```
*(You can also explicitly specify your port: `python 8-gesture-controller/dashboard.py --port /dev/tty.usbserial-XXXX`)*

## 🧠 Supported Gestures
- **Raise Up** / **Push Down** (Controls Volume)
- **Swipe Left** / **Swipe Right** (Previous / Next Track)
- **Push Forward** (Play / Pause)
- **Pull Back** (Mute / Unmute)
- **Roll Left** / **Roll Right** (Skip backward / forward 10s)
- **Clap** (Wakes the system up to listen for a command)

## 📚 Technical Evolution
Curious how the DSP and ML architectures were optimized to eliminate 4-second latency constraints? Read the full engineering breakdown in [TECHNICAL_CHRONOLOGY.md](TECHNICAL_CHRONOLOGY.md).