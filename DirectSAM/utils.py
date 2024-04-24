import cv2
import numpy as np
import torch
from torch import nn
import math
from itertools import product
import matplotlib.pyplot as plt
import numpy as np


def generate_crop_boxes(im_size, n_layers=1, overlap=0):
    """
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    """

    crop_boxes, layer_idxs = [], []
    im_w , im_h = im_size

    # Original image
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        # overlap = int(overlap_ratio * min(im_h, im_w) * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    # only keep layer_id=n_layers
    crop_boxes = [box for box, layer in zip(crop_boxes, layer_idxs) if layer == n_layers]
    layer_idxs = [layer for layer in layer_idxs if layer == n_layers]

    return crop_boxes


def boundary_zero_padding(probs, p=15):
    # from https://arxiv.org/abs/2308.13779
    
    zero_p = p//3
    alpha_p = zero_p*2
    
    probs[:, :alpha_p] *= 0.5
    probs[:, -alpha_p:] *= 0.5
    probs[:alpha_p, :] *= 0.5
    probs[-alpha_p:, :] *= 0.5

    probs[:, :zero_p] = 0
    probs[:, -zero_p:] = 0
    probs[:zero_p, :] = 0
    probs[-zero_p:, :] = 0

    return probs


def inference_single_image(image, image_processor, model, pyramid_layers=0, overlap=90, resolution=None):

    if resolution:
        image_processor.size['height'] = resolution
        image_processor.size['width'] = resolution

    def run(image, bzp=0):
        encoding = image_processor(image, return_tensors="pt")
        pixel_values = encoding.pixel_values.to(model.device).to(model.dtype)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        logits = outputs.logits.float().cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        probs = torch.sigmoid(upsampled_logits)[0, 0].detach().numpy()

        if bzp>0:
            probs = boundary_zero_padding(probs, p=bzp)
        return probs
    
    global_probs = run(image)

    if pyramid_layers > 0:
        for layer in range(1, pyramid_layers+1):
            boxes = generate_crop_boxes(image.size, n_layers=layer, overlap=overlap)
            for box in boxes:
                x1, y1, x2, y2 = box
                crop = image.crop(box)
                probs = run(crop, bzp=overlap)
                global_probs[y1:y2, x1:x2] += probs
        global_probs /= (pyramid_layers + 1)

    return global_probs


def probs_to_masks(probs, threshold=0.1):
        
    binarilized = (probs < threshold).astype(np.uint8)
    num_objects, labels = cv2.connectedComponents(binarilized)
    masks = [labels == i for i in range(1, labels.max() + 1)]
    masks.sort(key=lambda x: x.sum(), reverse=True)
    return masks


def visualize_masks(image, masks):
    canvas = np.ones_like(image) * 255

    for i in range(len(masks)):
        mask = masks[i]
        color = np.mean(image[mask], axis=0)
        canvas[mask] = color
    return canvas


def resize_to_max_length(image, max_length):
    width, height = image.size
    if width > height:
        new_width = max_length
        new_height = int(height * (max_length / width))
    else:
        new_height = max_length
        new_width = int(width * (max_length / height))
    return image.resize((new_width, new_height))


def visualize_direct_sam_result(probs, image, show_reconstruction=True, threshold=0.01, mask_cutoff = 256):

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(probs, cmap='PuBuGn')
    plt.title('Boundary Probabilities')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(probs > threshold, cmap='PuBuGn')
    plt.title(f'Binarilized Boundary Prediction (threshold={threshold})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    if show_reconstruction:
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f'Input Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        masks = probs_to_masks(probs, threshold=threshold)
        masks = sorted(masks, key=lambda x: np.sum(x), reverse=True)[:mask_cutoff]

        segment_visualization = visualize_masks(np.array(image), masks)
        plt.imshow(segment_visualization)

        plt.title(f'{len(masks)} Subobject Segments')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
