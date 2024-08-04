import cv2
import numpy as np
from PIL import Image
import random

def random_resize(image, scale_range=(0.75, 1.0)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    new_size = (int(image.width * scale), int(image.height * scale))
    return image.resize(new_size, Image.LANCZOS)

def resolution_down(image, max_resolution=(100, 100)):
    original_width, original_height = image.size
    scale = min(max_resolution[0] / original_width, max_resolution[1] / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    return image.resize((new_width, new_height), Image.LANCZOS)

def add_noise(image_pil: Image.Image, noise_level: float = 25.0) -> Image.Image:
    image_np = np.array(image_pil)
    noise = np.random.normal(0, noise_level, image_np.shape)
    noisy_image_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)
    noisy_image_pil = Image.fromarray(noisy_image_np, 'RGBA')
    return noisy_image_pil

def add_random_occlusions(image_pil: Image.Image, occlusion_images: list, num_occlusions: int = 3) -> Image.Image:
    # Convert to RGBA if not already in that mode
    if image_pil.mode != 'RGBA':
        image_pil = image_pil.convert('RGBA')
    # Copy the original image to avoid modifying it directly
    result_image = image_pil.copy()
    for _ in range(num_occlusions):
        # Randomly choose an occlusion image
        occlusion_pil = random.choice(occlusion_images)
        # Resize occlusion image if needed (optional)
        # Example: Resize occlusion to be up to 25% of the main image's width and height
        max_width, max_height = result_image.size
        occlusion_width, occlusion_height = occlusion_pil.size
        scale_factor = random.uniform(0.1, 0.25)
        new_width = int(max_width * scale_factor)
        new_height = int(occlusion_height * new_width / occlusion_width)
        occlusion_pil = occlusion_pil.resize((new_width, new_height), Image.ANTIALIAS)
        # Random position for the occlusion
        x = random.randint(0, max_width - new_width)
        y = random.randint(0, max_height - new_height)
        # Create a mask for the occlusion image
        mask = occlusion_pil.convert('L').point(lambda x: min(x, 200))
        # Paste the occlusion image onto the result image
        result_image.paste(occlusion_pil, (x, y), mask)
    return result_image
# Example usage:
# main_image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
# occlusion_images = [Image.open('path/to/occlusion1.png').convert('RGBA'), 
#                     Image.open('path/to/occlusion2.png').convert('RGBA')]
# occluded_image_pil = add_random_occlusions(main_image_pil, occlusion_images)
# occluded_image_pil.show()  # Display the image with occlusions

def downscale_and_upscale_image(pil_image, scale_factor):
    original_width, original_height = pil_image.size
    new_width = int(original_width / scale_factor)
    new_height = int(original_height / scale_factor)
    downscaled_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    upscaled_image = downscaled_image.resize((original_width, original_height), Image.LANCZOS)
    
    return upscaled_image

def random_perspective(image, max_warp=0.30):
    width, height = image.size
    pts1 = np.float32([[0,0], [width-1,0], [0,height-1], [width-1,height-1]])
    pts2 = np.float32([
        [np.random.uniform(0, max_warp * width), np.random.uniform(0, max_warp * height)],
        [width-1 - np.random.uniform(0, max_warp * width), np.random.uniform(0, max_warp * height)],
        [np.random.uniform(0, max_warp * width), height-1 - np.random.uniform(0, max_warp * height)],
        [width-1 - np.random.uniform(0, max_warp * width), height-1 - np.random.uniform(0, max_warp * height)]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    image_np = np.array(image)
    image_warped = cv2.warpPerspective(image_np, matrix, (width, height))
    return Image.fromarray(image_warped)

def random_rotation(image, angle_range=(-30, 30)):
    angle = np.random.uniform(angle_range[0], angle_range[1])
    return image.rotate(angle, expand=True)

def augment_logo(image):
    image = resolution_down(image)
    # image = random_resize(image)
    # image = random_perspective(image)
    # image = random_rotation(image)

    # image = add_noise(image)

    image = downscale_and_upscale_image(image,1.5)
    return image
