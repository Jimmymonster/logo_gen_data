import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import os

def random_resize(image, scale_range=(0.9, 1.5)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    new_size = (int(image.width * scale), int(image.height * scale))
    return image.resize(new_size, Image.LANCZOS)

def resolution_down(image, max_resolution=(80, 80)):
    original_width, original_height = image.size
    scale = min(max_resolution[0] / original_width, max_resolution[1] / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    return image.resize((new_width, new_height), Image.LANCZOS)

def add_noise(image_pil: Image.Image, min_noise_level: float = 25.0, max_noise_level: float = 75.0) -> Image.Image:
    image_np = np.array(image_pil)
    noise_level = random.uniform(min_noise_level, max_noise_level)
    noise = np.random.normal(0, noise_level, image_np.shape)
    noisy_image_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)
    noisy_image_pil = Image.fromarray(noisy_image_np, 'RGBA')
    return noisy_image_pil

def add_random_occlusions(image_pil: Image.Image, occlusion_images: list, num_occlusions: int = 3) -> Image.Image:
    if image_pil.mode != 'RGBA':
        image_pil = image_pil.convert('RGBA')
    result_image = image_pil.copy()
    max_width, max_height = result_image.size
    for _ in range(num_occlusions):
        # Randomly choose an occlusion image
        occlusion_pil = random.choice(occlusion_images)
        # Resize occlusion image if needed
        scale_factor = random.uniform(0.5, 1.0)
        occlusion_width, occlusion_height = occlusion_pil.size
        new_width = int(max_width * scale_factor)
        new_height = int(occlusion_height * new_width / occlusion_width)
        occlusion_pil = occlusion_pil.resize((new_width, new_height), Image.LANCZOS)
        center_x = max_width / 2
        center_y = max_height / 2
        margin = 20
        # Define the bounds to avoid placing the occlusion in the center area
        min_x = -new_width // 2
        max_x = max_width - new_width // 2
        min_y = -new_height // 2
        max_y = max_height - new_height // 2
        while True:
            x = random.randint(min_x, max_x)
            y = random.randint(min_y, max_y)
            # Check if the occlusion is within the margin area around the center
            occlusion_center_x = x + new_width / 2
            occlusion_center_y = y + new_height / 2
            if not (center_x - margin < occlusion_center_x < center_x + margin and
                    center_y - margin < occlusion_center_y < center_y + margin):
                break
        
        # Add mask version
        # mask = occlusion_pil.convert('L').point(lambda x: min(x, 200))
        # result_image.paste(occlusion_pil, (x, y), mask)

        # non mask version
        result_image.paste(occlusion_pil, (x, y), occlusion_pil)

    return result_image

def downscale_and_upscale_image(pil_image, scale_factor):
    original_width, original_height = pil_image.size
    new_width = int(original_width / scale_factor)
    new_height = int(original_height / scale_factor)
    downscaled_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    upscaled_image = downscaled_image.resize((original_width, original_height), Image.LANCZOS)
    return upscaled_image

def random_perspective(image, max_warp=0.2):
    width, height = image.size
    pts1 = np.float32([[0,0], [width-1,0], [0,height-1], [width-1,height-1]])
    pts2 = np.float32([
        [np.random.uniform(0, max_warp * width), np.random.uniform(0, max_warp * height)],
        [width-1 - np.random.uniform(0, max_warp * width), np.random.uniform(0, max_warp * height)],
        [np.random.uniform(0, max_warp * width), height-1 - np.random.uniform(0, max_warp * height)],
        [width-1 - np.random.uniform(0, max_warp * width), height-1 - np.random.uniform(0, max_warp * height)]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    
    # Apply perspective transform to the image
    image_np = np.array(image)
    warped_image = cv2.warpPerspective(image_np, matrix, (width, height))
    
    # Convert to grayscale and find non-zero points
    gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    non_zero_points = cv2.findNonZero(gray_warped)
    
    if non_zero_points is None:
        return image  # No valid transformation, return original image

    # Compute the bounding box
    x_min, y_min = np.min(non_zero_points, axis=0).flatten()
    x_max, y_max = np.max(non_zero_points, axis=0).flatten()

    # Ensure coordinates are within bounds
    x_min, y_min = int(max(0, x_min)), int(max(0, y_min))
    x_max, y_max = int(min(width - 1, x_max)), int(min(height - 1, y_max))

    # Crop the warped image to the bounding box
    cropped_warped_image = warped_image[y_min:y_max+1, x_min:x_max+1]
    result_image = Image.fromarray(cropped_warped_image)
    return result_image

def random_rotation(image, angle_range=(-60, 60)):
    angle = np.random.uniform(angle_range[0], angle_range[1])
    width, height = image.size

    # Calculate the new dimensions after rotation
    angle_rad = np.radians(angle)
    cos_a = np.abs(np.cos(angle_rad))
    sin_a = np.abs(np.sin(angle_rad))
    
    new_width = int(width * cos_a + height * sin_a)
    new_height = int(width * sin_a + height * cos_a)
    
    # Rotate the image and expand the canvas
    rotated_image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    # Resize rotated image to the calculated dimensions
    resized_image = rotated_image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
    
    return resized_image
    
def load_images_from_directory(directory_path: str) -> list:
    occlusion_images = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                image = Image.open(file_path).convert('RGBA')
                occlusion_images.append(image)
            except Exception as e:
                print(f"Failed to load image {file_path}: {e}")
    return occlusion_images

def random_stretch(image_pil: Image.Image, min_scale: float = 0.5, max_scale: float = 1.5) -> Image.Image:
    original_width, original_height = image_pil.size
    scale_x=1.0
    scale_y=1.0
    if(random.randint(0, 1) != 0):
        while(scale_x>0.9 and scale_x<1.1):
            scale_x = random.uniform(min_scale, max_scale)
    else:
        while(scale_y>0.9 and scale_y<1.1):
            scale_y = random.uniform(min_scale, max_scale)
    new_width = int(original_width * scale_x)
    new_height = int(original_height * scale_y)
    stretched_image = image_pil.resize((new_width, new_height), Image.Resampling.BILINEAR)
    return stretched_image

def random_color(img: Image.Image) -> Image.Image:
    if img.mode not in ('RGB', 'RGBA'):
        raise ValueError("Image mode must be RGB or RGBA")
    # Separate alpha channel if image is RGBA
    alpha = None
    if img.mode == 'RGBA':
        img, alpha = img.convert('RGB'), img.getchannel('A')
    # Randomly adjust brightness
    brightness_factor = random.uniform(0.95, 1.05)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    # Randomly adjust contrast
    contrast_factor = random.uniform(0.95, 1.05)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    # Randomly adjust hue and saturation
    hue_factor = random.uniform(-0.1, 0.1)
    saturation_factor = random.uniform(0.95, 1.05)
    def adjust_hue_saturation(image, hue_factor, saturation_factor):
        # Convert image to HSV mode
        hsv_image = image.convert('HSV')
        h, s, v = hsv_image.split()
        # Ensure the channels are in the correct mode
        if h.mode != 'L' or s.mode != 'L' or v.mode != 'L':
            raise ValueError("HSV channels are not in the correct mode")
        # Adjust saturation
        enhancer = ImageEnhance.Color(hsv_image)
        s = enhancer.enhance(saturation_factor).split()[1]
        # Adjust hue
        h = h.point(lambda p: (p + hue_factor * 255) % 255)
        hsv_image = Image.merge('HSV', (h, s, v))
        return hsv_image.convert('RGB')
    img = adjust_hue_saturation(img, hue_factor, saturation_factor)
    # Randomly adjust gamma
    gamma_factor = random.uniform(0.5, 2.0)
    img = ImageEnhance.Brightness(img).enhance(gamma_factor)
    # Reattach alpha channel if it was separated
    if alpha:
        img = Image.merge('RGBA', (img.convert('RGB').split() + (alpha,)))
    return img

def augment_logo(image,frame_range,logoIndex):
    image = resolution_down(image, (200,200))

    # image = random_resize(image)

    # if(random.randint(0, 9) != 0):
    # if(logoIndex < frame_range*0.9):
    #     image = random_perspective(image)
        
    # if(random.randint(0, 4) == 0):
    #     image = random_rotation(image)

    # if(logoIndex < frame_range*0.95):
    #     image = add_noise(image)

    # if(logoIndex < frame_range*0.90):
    #     occlusion_images = load_images_from_directory("occulusion_images")
    #     image = add_random_occlusions(image, occlusion_images, 1)

    # if(random.randint(0, 4) == 0):
    # if(logoIndex < frame_range*0.95):
    #     image = random_stretch(image,0.5,1.5)

    # if(logoIndex < frame_range*0.95):
    #     image = downscale_and_upscale_image(image,random.uniform(2, 4))
    # else:
    #     image = downscale_and_upscale_image(image,1.5)

    if(logoIndex < frame_range*0.90):
        # image = random_resize(image)
        image = random_perspective(image,0.2)
        # image = random_rotation(image)
        image = add_noise(image,25.0,75.0)
        # occlusion_images = load_images_from_directory("occulusion_images")
        # image = add_random_occlusions(image, occlusion_images, 1)
        image = random_stretch(image,0.5,1.5)

        image = random_color(image)
    image = downscale_and_upscale_image(image,1.5)

    return image
