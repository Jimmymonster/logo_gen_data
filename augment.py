import cv2
import numpy as np
from PIL import Image

def random_resize(image, scale_range=(0.8, 1.2)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    new_size = (int(image.width * scale), int(image.height * scale))
    return image.resize(new_size, Image.LANCZOS)

def resolution_down(image, max_resolution=(800, 800)):
    if image.width > max_resolution[0] or image.height > max_resolution[1]:
        new_size = (
            min(image.width, max_resolution[0]),
            min(image.height, max_resolution[1])
        )
        return image.resize(new_size, Image.LANCZOS)
    return image

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
    image_np = np.array(image)
    image_warped = cv2.warpPerspective(image_np, matrix, (width, height))
    return Image.fromarray(image_warped)

def random_rotation(image, angle_range=(-30, 30)):
    angle = np.random.uniform(angle_range[0], angle_range[1])
    return image.rotate(angle, expand=True)

def augment_logo(image):
    # image = random_resize(image)
    # image = resolution_down(image)
    image = random_perspective(image)
    image = random_rotation(image)
    return image
