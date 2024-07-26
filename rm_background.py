import os
import shutil
from rembg import remove
from PIL import Image

# Paths to your folders
logo_folder = 'logo'
old_images_folder = 'old_images'
if not os.path.exists(old_images_folder):
    os.makedirs(old_images_folder)

# Process each logo file
for logo_file in os.listdir(logo_folder):
    logo_path = os.path.join(logo_folder, logo_file)
    if os.path.isfile(logo_path):
        old_image_path = os.path.join(old_images_folder, logo_file)
        shutil.copyfile(logo_path, old_image_path)
        with Image.open(logo_path) as image:
            transparent_image = remove(image)
            new_logo_path = os.path.join(logo_folder, os.path.splitext(logo_file)[0] + '.png')
            os.remove(os.path.join(logo_folder, logo_file))
            transparent_image.save(new_logo_path, format='PNG')
        print(f"Processed and saved {logo_file} with transparent background. Moved original to {old_image_path}")

print("Background removal complete.")
