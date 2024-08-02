import cv2
import numpy as np
from PIL import Image

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
    # image = random_resize(image)
    image = resolution_down(image)
    # image = random_perspective(image)
    # image = random_rotation(image)
    image = downscale_and_upscale_image(image,1.5)
    return image
