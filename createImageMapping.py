import os
import json

def extract_name_mapping_from_label_studio(json_file):
    name_mapping = {}
    first = True
    with open(json_file, 'r') as f:
        data = json.load(f)
        for item in data:
            old_name = item['data']['image'].split('/')[-1].split('-',1)[1]
            new_name = item['file_upload']
            name_mapping[old_name] = new_name
            if first:
                project_num = item['data']['image'].split('/')[-2]
    # print(name_mapping)
    return name_mapping,project_num

def convert_yolo_to_label_studio(yolo_folder, name_mapping, output_file):
    images_folder = os.path.join(yolo_folder, 'images')
    labels_folder = os.path.join(yolo_folder, 'labels')
    
    # Read classes
    classes_file = os.path.join(yolo_folder, 'classes.txt')
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    combined_data = []

    # Process each label file
    for label_file in os.listdir(labels_folder):
        if label_file.endswith('.txt'):
            old_image_name = os.path.splitext(label_file)[0] + '.png'
            new_image_name = name_mapping[old_image_name]
            image_path = os.path.join(images_folder, old_image_name)
            label_path = os.path.join(labels_folder, label_file)

            # Read label file
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # Convert to Label Studio format
            annotations = []
            for line in lines:
                parts = line.strip().split()
                class_id, x_center, y_center, width, height = map(float, parts)
                x_center *= 100
                y_center *= 100
                width *= 100
                height *= 100
                
                annotation = {
                    # "original_width": 704,
                    # "original_height": 576,
                    # "image_rotation": 0,
                    "type": "rectanglelabels",
                    "value": {
                        "x": x_center - width / 2,
                        "y": y_center - height / 2,
                        "width": width,
                        "height": height,
                        # "image_rotation": 0,
                        "rectanglelabels": [classes[int(class_id)]]
                    },
                    "from_name": "label",
                    "to_name": "image",
                    "origin": "manual",
                }
                annotations.append(annotation)

            # Prepare data
            data = {
                "data": {
                    "image": f"/data/upload/{project_num}/{new_image_name}"
                },
                "annotations": [
                    {
                        "result": annotations,
                        "ground_truth": "False",
                        # "updated_by": "3",
                    }
                ]
            }
            combined_data.append(data)

    # Save to a single JSON file
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)

# Usage
yolo_folder = 'output'
label_studio_json_file = 'test.json'
output_file = 'combined_annotations.json'

# Extract name mapping from Label Studio JSON
name_mapping , project_num = extract_name_mapping_from_label_studio(label_studio_json_file)

# Convert YOLO annotations to Label Studio format and save
convert_yolo_to_label_studio(yolo_folder, name_mapping, output_file)
