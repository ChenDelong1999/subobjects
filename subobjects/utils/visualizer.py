import torch
import cv2
from PIL import Image
import numpy as np
import tqdm
import matplotlib.pyplot as plt


def visualize_masks(image, masks, mask_erotion=0, show_progress_bar=False, random_colors=False):
    if type(image) == Image.Image:
        image = np.array(image)
    canvas = np.ones_like(image) * 255

    # sort masks by area
    masks = sorted(masks, key=lambda x: np.sum(x), reverse=True)

    masks = tqdm.tqdm(masks) if show_progress_bar else masks
    for mask in masks:
        if mask_erotion>0:
            kernel = np.ones((mask_erotion, mask_erotion), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel)
        mask_indices = np.where(mask)
        if random_colors:
            color = np.random.randint(0, 255, size=(3,))
        else:
            color = np.mean(image[mask_indices], axis=0)
        canvas[mask_indices] = color
    return canvas


def visualize_sample(sample, inputs):
    print(sample['text'])

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(inputs['image'][0])

    plt.subplot(1, 2, 2)
    plt.imshow(visualize_masks(inputs['image'][0], inputs['masks'][0].cpu().numpy()))
    plt.show()

def label_map_to_random_rgb(label_map, seed=None):
    """
    Convert a (H, W) integer label map into a (H, W, 3) random RGB image.

    Args:
        label_map (torch.Tensor or np.ndarray): Single-channel label map of shape (H, W).
        seed (int, optional): If provided, sets a random seed for reproducible colors.

    Returns:
        np.ndarray: Color image of shape (H, W, 3) with dtype=np.uint8.
    """
    # If user wants reproducible colors, set the seed
    if seed is not None:
        np.random.seed(seed)

    # Convert label_map to a CPU tensor (if torch) or ndarray
    if isinstance(label_map, torch.Tensor):
        label_map = label_map.cpu().numpy()
    label_map = label_map.astype(np.int32)

    # Determine the number of classes (maximum label)
    max_label = label_map.max()
    if max_label < 1:
        # If there are no labels > 0, just return a blank image
        h, w = label_map.shape
        return np.zeros((h, w, 3), dtype=np.uint8)

    # Generate random colors for each label: shape (max_label+1, 3)
    #   e.g., color 0 = black, color 1 = random, color 2 = random, etc.
    #   If you want label 0 to also have a random color, just remove the special case for 0.
    colors = np.zeros((max_label + 1, 3), dtype=np.uint8)
    for lbl in range(1, max_label + 1):
        colors[lbl] = np.random.randint(0, 256, size=3)

    # Map each label to its corresponding color
    rgb_image = colors[label_map]

    return rgb_image