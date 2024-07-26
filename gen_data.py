import cv2
import numpy as np
import os
import random
import shutil

logo_folder = 'logo'
video_folder = 'video'
result_folder = 'result'
num_snapshots = 100
padding = 10

def clear_result_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
clear_result_folder(result_folder)

logo_files = [f for f in os.listdir(logo_folder) if os.path.isfile(os.path.join(logo_folder, f))]
if not logo_files:
    print("No logo files found in the logo folder.")
    exit()

video_files = [f for f in os.listdir(video_folder) if os.path.isfile(os.path.join(video_folder, f))]
if not video_files:
    print("No video files found in the video folder.")
    exit()

video_path = os.path.join(video_folder, video_files[0])
video_capture = cv2.VideoCapture(video_path)

ret, frame = video_capture.read()
if not ret:
    print("Failed to capture video frame.")
    video_capture.release()
    # cv2.destroyAllWindows()
    exit()

frame_height, frame_width = frame.shape[:2]
def place_logo_and_crop(frame, logo, padding):
    logo_height, logo_width = logo.shape[:2]
    if logo.shape[2] == 4:
        alpha_channel = logo[:, :, 3] / 255.0
        logo_rgb = logo[:, :, :3]
    else:
        alpha_channel = np.ones((logo_height, logo_width))
        logo_rgb = logo
    x = random.randint(0, frame_width - logo_width)
    y = random.randint(0, frame_height - logo_height)
    roi = frame[y:y+logo_height, x:x+logo_width]
    for c in range(0, 3):
        roi[:, :, c] = (alpha_channel * logo_rgb[:, :, c] + (1 - alpha_channel) * roi[:, :, c])
    frame[y:y+logo_height, x:x+logo_width] = roi
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + logo_width + padding, frame_width)
    y2 = min(y + logo_height + padding, frame_height)
    cropped_logo_area = frame[y1:y2, x1:x2]
    return cropped_logo_area
for i in range(num_snapshots):
    logo_path = os.path.join(logo_folder, random.choice(logo_files))
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    frame_number = random.randint(0, int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video_capture.read()
    if not ret:
        print(f"Failed to capture video frame at position {frame_number}.")
        continue
    cropped_logo_area = place_logo_and_crop(frame.copy(), logo, padding)
    timestamp = int(video_capture.get(cv2.CAP_PROP_POS_MSEC)) // 1000  # Convert milliseconds to seconds
    result_path = os.path.join(result_folder, f'snapshot_{i+1}_frame_{frame_number}_timestamp_{timestamp}s.jpg')
    cv2.imwrite(result_path, cropped_logo_area)
    print(f"Snapshot {i+1} saved to {result_path}")
video_capture.release()
# cv2.destroyAllWindows()
