import os
import pyautogui
import keyboard

folder_path = "dataset"
os.makedirs(folder_path, exist_ok=True)

def get_next_filename():
    files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    numbers = []

    for f in files:
        try:
            numbers.append(int(os.path.splitext(f)[0]))
        except:
            pass

    return os.path.join(folder_path, f"{max(numbers)+1 if numbers else 1}.png")
# base warps hexes bridges shapes
def take_screenshot():
    filepath = get_next_filename()
    
    # region = (left, top, width, height)
    screenshot = pyautogui.screenshot(region=(1060, 87, 1965 - 1060, 1682 - 87))
    screenshot.save(filepath)

    print(f"Saved: {filepath}")

# bind keys
keyboard.add_hotkey("F8", take_screenshot)

print("Running... Press F8 to capture, F9 to quit.")

# keep program alive
keyboard.wait()