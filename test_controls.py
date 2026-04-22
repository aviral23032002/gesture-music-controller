from pynput.keyboard import Key, Controller
import time
import sys

# Initialize the keyboard controller
keyboard = Controller()

def execute_command(gesture):
    print(f"\n>>> EXECUTING: {gesture.upper()} <<<")
    
    if gesture == 'up':
        # Taps the global volume up key
        keyboard.tap(Key.media_volume_up)
        print("Action: System Volume Up")
        
    elif gesture == 'down':
        # Taps the global volume down key
        keyboard.tap(Key.media_volume_down)
        print("Action: System Volume Down")
        
    elif gesture == 'rotate_right':
        # Taps the global "Next Track" media key
        keyboard.tap(Key.media_next)
        print("Action: Global Next Track")
        
    elif gesture == 'rotate_left':
        # Taps the global "Previous Track" media key
        keyboard.tap(Key.media_previous)
        print("Action: Global Previous Track")
    
    elif gesture == 'shake':
        # Taps the global "Play/Pause" media key
        keyboard.tap(Key.media_play_pause)
        print("Action: Global Play/Pause")
    
    elif gesture == 'push':
        # Taps the global "Volume Mute" media key
        keyboard.tap(Key.media_volume_mute)
        print("Action: Global Volume Mute")
        
    else:
        print("Unknown command. Try 'up', 'down', 'rotate_right', or 'rotate_left'.")

def main():
    print("--- Global Media Control Tester ---")
    print("This controls whatever audio is currently playing on your Mac.")
    print("-" * 37)

    try:
        while True:
            user_input = input("\nEnter gesture: ").strip().lower()
            
            if user_input in ['exit', 'quit']:
                break
                
            execute_command(user_input)
            time.sleep(0.5) 
            
    except KeyboardInterrupt:
        print("\nExiting tester...")
        sys.exit()

if __name__ == "__main__":
    main()