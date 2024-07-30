import cv2
import numpy as np
import os
import random
from PIL import Image
import shutil
from augment import augment_logo

# Paths to your folders
video_path = 'video/video.mp4'
logo_folder = 'logo'
output_folder = 'output'
classes_file = 'classes.txt'
class_list = ['logo']
labels_folder = 'labels'
images_folder = 'images'

# Create output directories
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(os.path.join(output_folder, labels_folder)):
    os.makedirs(os.path.join(output_folder, labels_folder))
if not os.path.exists(os.path.join(output_folder, images_folder)):
    os.makedirs(os.path.join(output_folder, images_folder))

# Load logo
logo_files = os.listdir(logo_folder)


# Extract frames from video and overlay logo
def overlay_logo(frame, logo):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")

    aug_logo = augment_logo(logo)
    logo_width, logo_height = aug_logo.size
    
    # Random position with padding
    # x_padding = random.randint(5, 10)  # Random padding on x-axis
    # y_padding = random.randint(5, 10)  # Random padding on y-axis
    # random_pos_x = random.randint(0,x_padding)
    # random_pos_y = random.randint(0,y_padding)
    x_padding = 0  
    y_padding = 0  
    random_pos_x = 0
    random_pos_y = 0
    x = random.randint(x_padding, frame.shape[1] - logo_width - x_padding)
    y = random.randint(y_padding, frame.shape[0] - logo_height - y_padding)
    
    # Create a transparent background
    transparent_frame = Image.new('RGBA', frame_pil.size, (0, 0, 0, 0))
    transparent_frame.paste(frame_pil, (0, 0))
    
    # Overlay the logo
    transparent_frame.paste(aug_logo, (x, y), aug_logo)
    return cv2.cvtColor(np.array(transparent_frame), cv2.COLOR_RGBA2BGR), (x-random_pos_x, y-random_pos_y, logo_width + x_padding, logo_height + y_padding )

# Process video
cap = cv2.VideoCapture(video_path)
frame_number = 0
snapshot_interval = 300  # Capture one frame every 30 frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_number % snapshot_interval == 0:
        logo_file=logo_files[random.randint(0,len(logo_files)-1)]
        logo_path = os.path.join(logo_folder, logo_file)
        logo = Image.open(logo_path).convert("RGBA")

        # Overlay logo and save snapshot
        processed_frame, bbox = overlay_logo(frame, logo)
        snapshot_filename = f"frame_{frame_number}.jpg"
        snapshot_path = os.path.join(output_folder, images_folder, snapshot_filename)
        cv2.imwrite(snapshot_path, processed_frame)
        
        # Write labels (YOLO format: class_id x_center y_center width height)
        label_filename = os.path.splitext(snapshot_filename)[0] + '.txt'
        label_path = os.path.join(output_folder, labels_folder, label_filename)
        h, w, _ = frame.shape
        x_center = (bbox[0] + bbox[2] / 2) / w
        y_center = (bbox[1] + bbox[3] / 2) / h
        width = bbox[2] / w
        height = bbox[3] / h
        
        with open(label_path, 'w') as label_file:
            label_file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")  # Example label

    frame_number += 1

cap.release()

# Create classes.txt
with open(os.path.join(output_folder, classes_file), 'w') as class_file:
    for item in class_list:
        class_file.write(f"{item}\n")  

print("Video processing complete. Output saved in the 'output' folder.")
