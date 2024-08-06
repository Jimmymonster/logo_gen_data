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

# Load logos from subfolders
logo_folders = [f.path for f in os.scandir(logo_folder) if f.is_dir()]
class_list = []

logo_files = []
for i, folder in enumerate(logo_folders):
    class_name = os.path.basename(folder)
    class_list.append(class_name)
    logo_files.extend([(folder, file) for file in os.listdir(folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Function to check if a logo overlaps with existing logos
def is_overlapping(new_bbox, existing_bboxes):
    nx, ny, nw, nh = new_bbox
    for (ex, ey, ew, eh) in existing_bboxes:
        if (nx < ex + ew and nx + nw > ex and ny < ey + eh and ny + nh > ey):
            return True
    return False

# Overlay logos on frame
def overlay_logos(frame, logos, max_logos=3):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    existing_bboxes = []
    bboxes = []
    class_ids = []
    global frame_range
    global logoIndex

    for _ in range(min(max_logos, len(logos))):
        logo_folder, logo_file = random.choice(logo_files)
        logo_class_id = class_list.index(os.path.basename(logo_folder))
        logo_path = os.path.join(logo_folder, logo_file)
        logo = Image.open(logo_path).convert("RGBA")
        aug_logo = augment_logo(logo,frame_range,logoIndex)
        logoIndex+=1
        logo_width, logo_height = aug_logo.size
        
        # x_padding = random.randint(5, 10)  
        # y_padding = random.randint(5, 10)  
        # random_pos_x = random.randint(0, 5)
        # random_pos_y = random.randint(0, 5)

        x_padding = 0
        y_padding = 0 
        random_pos_x = 0
        random_pos_y = 0
        
        # Try to find a non-overlapping position
        for _ in range(100):  # Try 100 times to avoid infinite loops
            x = random.randint(x_padding, frame.shape[1] - logo_width - x_padding)
            y = random.randint(y_padding, frame.shape[0] - logo_height - y_padding)
            bbox = (x-random_pos_x, y-random_pos_y, logo_width + x_padding, logo_height + y_padding)
            
            if not is_overlapping(bbox, existing_bboxes):
                existing_bboxes.append(bbox)
                transparent_frame = Image.new('RGBA', frame_pil.size, (0, 0, 0, 0))
                transparent_frame.paste(frame_pil, (0, 0))
                transparent_frame.paste(aug_logo, (x, y), aug_logo)
                frame_pil = transparent_frame
                bboxes.append(bbox)
                class_ids.append(logo_class_id)
                break
    
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR), bboxes, class_ids

# Process video
cap = cv2.VideoCapture(video_path)
frame_number = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define range for random frame selection
frame_range = 50  # Number of frames to randomly select
logoIndex = 0
selected_frames = sorted(random.sample(range(total_frames), frame_range))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_number in selected_frames:
        # Prepare to overlay multiple logos
        # num_logos = random.randint(1, 3)
        num_logos = 1
        processed_frame, bboxes, class_ids = overlay_logos(frame, logo_files, num_logos)
        
        # Save snapshot
        snapshot_filename = f"frame_{frame_number}.png"
        snapshot_path = os.path.join(output_folder, images_folder, snapshot_filename)
        cv2.imwrite(snapshot_path, processed_frame)
        
        # Write labels (YOLO format: class_id x_center y_center width height)
        label_filename = os.path.splitext(snapshot_filename)[0] + '.txt'
        label_path = os.path.join(output_folder, labels_folder, label_filename)
        h, w, _ = frame.shape
        
        with open(label_path, 'w') as label_file:
            for bbox, class_id in zip(bboxes, class_ids):
                x_center = (bbox[0] + bbox[2] / 2) / w
                y_center = (bbox[1] + bbox[3] / 2) / h
                width = bbox[2] / w
                height = bbox[3] / h
                label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    frame_number += 1

cap.release()

# Create classes.txt
with open(os.path.join(output_folder, classes_file), 'w') as class_file:
    for item in class_list:
        class_file.write(f"{item}\n")

print("Video processing complete. Output saved in the 'output' folder.")
