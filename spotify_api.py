import subprocess

def forward_10_seconds():
    # This AppleScript tells the Spotify app to jump 10s ahead
    script = 'tell application "Spotify" to set player position to (player position + 10)'
    
    try:
        subprocess.run(['osascript', '-e', script], check=True)
        print("Jumped forward 10 seconds!")
    except Exception as e:
        print(f"Error: {e}. Make sure Spotify is open and playing.")

if __name__ == "__main__":
    forward_10_seconds()