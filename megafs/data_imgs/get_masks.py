# @title optimized version for mask collection
import os
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Mapping from integer label to mask name
str2num = {
    1: 'skin', 2: 'l_brow', 3: 'r_brow', 4: 'l_eye', 5: 'r_eye',
    7: 'l_ear', 8: 'r_ear', 10: 'nose', 11: 'mouth',
    12: 'u_lip', 13: 'l_lip', 14: 'neck', 17: 'hair'
}

# Output folder for storing masks
output_folder = "/masks_output"
os.makedirs(output_folder, exist_ok=True)


# Cache all file paths to speed up `find_path`
def build_file_index(dir_path="/CelebAMask-HQ/CelebAMask-HQ-mask-anno"):
    file_index = {}
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".png"):
                key = file.split('_')[0]  # Get image ID
                if key not in file_index:
                    file_index[key] = {}
                file_index[key][file] = os.path.join(root, file)
    return file_index


file_index = build_file_index()


def find_path(file_end, img_id):
    """Find the file path using the cached index."""
    return file_index.get(str(img_id), {}).get(f"{img_id}_{file_end}.png", None)


def load_array(path_to_file):
    """Load image as NumPy array and ensure correct shape."""
    if path_to_file and os.path.exists(path_to_file):
        im_frame = Image.open(path_to_file).convert('L')  # Convert to grayscale
        np_frame = np.array(im_frame)
        if np_frame.shape != (512, 512):  # Check for expected shape
            print(f"Warning: Unexpected shape {np_frame.shape} for {path_to_file}")
        return np_frame
    else:
        return np.zeros((512, 512), dtype=np.uint8)  # Return blank mask if missing


def encode_segmentation_rgb(tgt_idx, no_neck=True):
    """Generate segmentation mask for an image ID and save it."""
    face_part_ids = [1, 2, 3, 4, 5, 10, 12, 13] if no_neck else [1, 2, 3, 4, 5, 7, 8, 10, 12, 13, 14]

    mouth_map = load_array(find_path("mouth", tgt_idx))
    hair_map = load_array(find_path("hair", tgt_idx))
    face_map = np.zeros_like(mouth_map)

    for valid_id in face_part_ids:
        attr_path = find_path(str2num[valid_id], tgt_idx)
        if attr_path:
            face_map += load_array(attr_path)

    # Clip to valid range and stack as a segmentation mask
    face_map = np.clip(face_map, 0, 255).astype(np.uint8)
    mouth_map = np.clip(mouth_map, 0, 255).astype(np.uint8)
    hair_map = np.clip(hair_map, 0, 255).astype(np.uint8)
    mask = np.stack([face_map, mouth_map, hair_map], axis=2)

    # Save mask as an image
    mask_image = Image.fromarray(mask)
    mask_image.save(os.path.join(output_folder, f"{tgt_idx}_mask.png"))

    # return mask


# Parallel Processing for Faster Execution
def process_batch(img_ids, use_gpu=False):
    """Process a batch of images with multiprocessing or GPU acceleration."""
    if use_gpu:
        device = torch.device("cuda")
        for idx in tqdm(img_ids):
            encode_segmentation_rgb(idx)
        # masks = [torch.tensor(encode_segmentation_rgb(idx)).to(device) for idx in img_ids]
        # return masks
    else:
        with ProcessPoolExecutor() as executor:
            # masks = list(executor.map(encode_segmentation_rgb, img_ids))
            executor.map(encode_segmentation_rgb, img_ids)
        # return masks


# Example usage: Process all available images
img_data_path = list(range(0, 30000))
process_batch(img_data_path, use_gpu=False)  # Set to True for GPU processing
