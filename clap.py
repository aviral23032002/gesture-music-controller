import pyaudio
import struct
import os

# --- CONFIGURATION ---
THRESHOLD = 0.5        # Sensitivity (Lower = more sensitive)
CHANNELS = 1
RATE = 44100
CHUNK = 1024
FORMAT = pyaudio.paInt16
# ---------------------

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening for a clap... (Press Ctrl+C to stop)")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        # Convert audio data to integers
        count = len(data) / 2
        format_str = "%dh" % count
        shorts = struct.unpack(format_str, data)
        
        # Get the peak volume of the chunk
        peak = max(shorts) / 32767.0 

        if peak > THRESHOLD:
            print(f"Clap detected! (Volume: {peak:.2f})")
            # This command opens Google Chrome on macOS
            os.system("open -a 'Google Chrome'")
            # Pause briefly to avoid multi-triggering from one clap
            import time
            time.sleep(2)
            
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()