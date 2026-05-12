# 🎵 Gesture Music Controller

A high-performance, real-time gesture recognition system built with an **ESP32**, an IMU sensor, and a **Python Machine Learning Backend**. Control Spotify or system audio entirely through mid-air gestures with zero perceived latency.

## 🚀 Features
- **Zero-Latency Signal Processing**: Implements software Butterworth Low-Pass and High-Pass filters to clean noisy IMU data before it reaches the AI.
- **Hybrid Flash-AI Architecture**: Combines a lightweight Machine Learning model with hardcoded physical-polarity gating to solve "sign-flip" errors, yielding **99%+ accuracy**!
- **Real-time 6-Panel Dashboard**: Built with Matplotlib, showing live waveform data, session history, confidence breakdown, and real-time Spotify metadata (track, artist, artwork, volume).
- **AppleScript Spotify Integration**: Seamlessly controls the macOS Spotify app via native AppleScript commands—no API keys required.

---

## 🛠️ Hardware Setup

### Wiring Guide
Connect your IMU sensor (e.g., MPU6050 / BNO085) to the ESP32 using the following I2C pins:

| IMU Pin | ESP32 Pin |
|---------|-----------|
| **SCL** | GPIO 19   |
| **SDA** | GPIO 18   |
| **VCC** | 3.3V      |
| **GND** | GND       |

---

## 💻 Getting Started

This project is split into two parts: the C/C++ firmware that runs on the ESP32, and the Python backend/dashboard that runs on your computer.

### Step 1: Flash the ESP32 Firmware
You need the **ESP-IDF** environment installed to build the C code. 

1. Open your terminal and activate your ESP-IDF export script (e.g., `. ~/esp/esp-idf/export.sh`).
2. Navigate to the root directory of this project.
3. Build and flash the firmware to your ESP32:

```bash
# Set target architecture
idf.py set-target esp32

# Build the project
idf.py build

# Flash and monitor the output (replace /dev/cu.usbserial-XXXX with your actual port)
idf.py -p /dev/cu.usbserial-XXXX flash monitor
```
*Note: The ESP32 will immediately begin broadcasting raw Accelerometer and Gyroscope data over Serial at 115200 baud.*

### Step 2: Setup the Python Dashboard
The machine learning pipeline and the visual dashboard live in the `8-gesture-controller/` directory.

1. Open a new terminal window in the root directory.
2. Create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install all required dependencies using the provided requirements file:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Controller
With your ESP32 plugged in and broadcasting data, start the dashboard!

```bash
python 8-gesture-controller/dashboard.py
```
*(If the script cannot automatically find your ESP32, you can specify the port manually: `python 8-gesture-controller/dashboard.py --port /dev/tty.usbserial-XXXX`)*

---

## 🧠 Supported Gestures

Once the dashboard is running, the system will sit in an **Idle** state.

1. **Clap**: Perform a sharp clap to "Wake" the system. It will listen for 1 second.
2. **Execute Command**: Within that 1-second window, perform one of the following gestures:
   - **Raise Up** / **Push Down**: Volume Up / Volume Down
   - **Swipe Left** / **Swipe Right**: Previous Track / Next Track
   - **Push Forward**: Play / Pause
   - **Pull Back**: Mute / Unmute
   - **Roll Left** / **Roll Right**: Skip backward 10s / Skip forward 10s

---

## 📚 Technical Evolution
Curious how the DSP and ML architectures were optimized to eliminate the massive latency constraints found in standard Human Activity Recognition models? Read the full engineering breakdown in [TECHNICAL_CHRONOLOGY.md](TECHNICAL_CHRONOLOGY.md).