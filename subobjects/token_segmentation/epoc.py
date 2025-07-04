# directsam.py
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from skimage.segmentation import watershed
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor

class EPOCSegmenter:
    def __init__(
        self,
        image_resolution=1024,
        checkpoint="chendelong/DirectSAM-b0-1024px-sa1b-2ep-1017",
        threshold=0.1,
        max_tokens=512,
        crop=1,
        device="cuda",
        **kwargs
    ):
        self.ckpt = checkpoint
        self.threshold = threshold
        self.image_resolution = image_resolution
        self.max_tokens = max_tokens
        self.crop = crop
        self.device = device

        self.image_processor = AutoImageProcessor.from_pretrained(
            "chendelong/DirectSAM-1800px-0424", reduce_labels=True
        )
        self.image_processor.size['height'] = image_resolution // crop
        self.image_processor.size['width'] = image_resolution // crop

        self.model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint)
        self.model = self.model.to(self.device).half().eval()

        print(f'DirectSAM initialized on {device}')

    def load_images(self, images):
        """
        Loads and preprocesses images.

        Args:
            images (str, np.ndarray, PIL.Image.Image, or list): Input images.

        Returns:
            list of PIL.Image.Image: Preprocessed images.
        """
        if not isinstance(images, list):
            images = [images]

        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif not isinstance(img, Image.Image):
                raise TypeError("Unsupported image type.")

            img = img.convert("RGB").resize((self.image_resolution, self.image_resolution))
            processed_images.append(img)
        return processed_images

    def split_images_into_patches(self, images):
        """
        Splits each image into self.crop x self.crop patches.

        Args:
            images (list of PIL.Image.Image): The list of images to split.

        Returns:
            tuple:
                patches (list of PIL.Image.Image): The list of image patches.
                positions (list of tuples): The positions of each patch in the original images.
        """
        patches = []
        positions = []
        for image_idx, image in enumerate(images):
            W, H = image.size
            patch_w = W // self.crop
            patch_h = H // self.crop
            for i in range(self.crop):
                for j in range(self.crop):
                    left = patch_w * j
                    upper = patch_h * i
                    right = left + patch_w
                    lower = upper + patch_h
                    box = (left, upper, right, lower)
                    patch = image.crop(box)
                    patches.append(patch)
                    positions.append((image_idx, i, j))
        return patches, positions

    @torch.inference_mode()
    def directsam_forward(self, patches):
        """
        Processes image patches through the model to obtain probabilities and boundaries.

        Args:
            patches (list of PIL.Image.Image): The list of image patches.

        Returns:
            tuple:
                probabilities (np.ndarray): The probabilities from the model output.
                boundaries (np.ndarray): The boundary maps obtained by thresholding the probabilities.
        """
        pixel_values = self.image_processor(patches, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device).to(self.model.dtype)

        logits = self.model(pixel_values=pixel_values).logits
        logits = logits.float().cpu()
        patch_size = self.image_resolution // self.crop
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=(patch_size, patch_size),
            mode="bicubic",
            align_corners=False,
        )

        probabilities = torch.sigmoid(upsampled_logits)[:, 0].numpy()
        boundaries = probabilities > self.threshold

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 10))
        # plt.imshow(probabilities[0], cmap='PuBu')
        # plt.axis('off')
        # plt.show()


        return probabilities, boundaries
    
    def reassemble_masks(self, probabilities, boundaries, positions):
        """
        Reassembles the masks from patches back into full-size images.

        Args:
            probabilities (np.ndarray): Probabilities from the model.
            boundaries (np.ndarray): Boundary maps for each patch.
            positions (list of tuples): Positions of each patch in the original images.

        Returns:
            dict: A dictionary mapping image indices to their list of masks with positions.
        """
        masks_dict = {image_idx: [] for image_idx, _, _ in positions}
        patch_h = self.image_resolution // self.crop
        patch_w = self.image_resolution // self.crop

        for idx, (image_idx, i, j) in enumerate(positions):
            boundary = boundaries[idx]
            prob = probabilities[idx]
            masks = self.boundary_to_mask(boundary, prob)
            top = i * patch_h
            left = j * patch_w
            for mask in masks:
                # Store the mask along with its position to avoid immediate expansion to full size
                masks_dict[image_idx].append({'mask': mask, 'top': top, 'left': left})
        return masks_dict

    def combine_and_sort_masks(self, masks_dict):
        """
        Combines masks for each image, sorts them by area, and ensures the number of masks does not exceed max_tokens.

        Args:
            masks_dict (dict): A dictionary mapping image indices to their list of masks with positions.

        Returns:
            torch.Tensor: The final batch of masks for all images.
        """
        batch_masks = []
        for image_idx in sorted(masks_dict.keys()):
            masks_info = masks_dict[image_idx]
            if len(masks_info) == 0:
                empty_masks = np.zeros(
                    (self.max_tokens, self.image_resolution, self.image_resolution), dtype=bool
                )
                batch_masks.append(empty_masks)
                continue

            # Extract masks and their positions
            masks = [info['mask'] for info in masks_info]
            tops = [info['top'] for info in masks_info]
            lefts = [info['left'] for info in masks_info]

            # Compute areas
            areas = np.array([np.sum(mask) for mask in masks])

            # Sort masks by area
            sorted_indices = np.argsort(areas)[::-1]

            # Select top max_tokens masks
            selected_indices = sorted_indices[:self.max_tokens]
            num_selected = len(selected_indices)

            # Prepare an array to hold the selected masks
            selected_masks = np.zeros(
                (self.max_tokens, self.image_resolution, self.image_resolution), dtype=bool
            )

            for idx_in_selected, idx in enumerate(selected_indices):
                mask = masks[idx]
                top = tops[idx]
                left = lefts[idx]
                h, w = mask.shape
                # Directly place the mask into the correct location
                selected_masks[idx_in_selected, top:top+h, left:left+w] = mask

            batch_masks.append(selected_masks)
        batch_masks = np.stack(batch_masks)
        batch_masks = torch.tensor(batch_masks)
        return batch_masks

    @torch.inference_mode()
    def __call__(self, images):
        import time
        """
        Processes images to generate masks.

        Args:
            images (str, np.ndarray, PIL.Image.Image, or list): Input images.

        Returns:
            torch.Tensor: Batch of masks for all images.
        """

        images = self.load_images(images)

        import time


        if self.crop > 1:
            patches, positions = self.split_images_into_patches(images)
            probabilities, boundaries = self.directsam_forward(patches)
            masks_dict = self.reassemble_masks(probabilities, boundaries, positions)
        else:
            start = time.time()
            probabilities, boundaries = self.directsam_forward(images)
            # print(f'Segformer time: {time.time()-start}')
            start = time.time()
            masks_dict = {}
            for idx in range(len(images)):
                boundary = boundaries[idx]
                prob = probabilities[idx]
                masks = self.boundary_to_mask(boundary, prob)
                # Store masks with positional information (top=0, left=0)
                masks_dict[idx] = [{'mask': mask, 'top': 0, 'left': 0} for mask in masks]
            # print(f'Watershed time:                          {time.time()-start}')

        batch_masks = self.combine_and_sort_masks(masks_dict)

        return batch_masks

    def boundary_to_mask(self, boundary, probability):
        """
        Converts a boundary image to a binary mask.
        Input:      A numpy array (H, W) representing the boundary image, True for boundary pixels and False for non-boundary pixels.
        Returns:    A numpy array of binary masks (n_masks, H, W), where each mask corresponds to a connected component in the boundary image.
        """
        if self.threshold<1:
            num_objects, markers = cv2.connectedComponents(
                (~boundary).astype(np.uint8), 
                connectivity=4, 
                )
        else:
            markers = self.threshold

        labels = watershed(
            probability, 
            markers=markers,
            )

        masks = np.zeros((
            # num_objects-1,
            np.max(labels)+1,
             *boundary.shape), dtype=bool)
        
        for i in range(1, np.max(labels)+1):
            masks[i-1] = labels == i

        # sort by area
        areas = np.sum(masks, axis=(1, 2))
        sorted_indices = np.argsort(areas)[::-1]
        masks = masks[sorted_indices]

        # # if there are more than max_tokens masks, merge the small masks
        # if num_objects > self.max_tokens + 1:
        #     remaining_masks = masks[self.max_tokens-1:]
        #     remaining_masks = np.any(remaining_masks, axis=0)
        #     masks[self.max_tokens-1] = remaining_masks
        return masks[:self.max_tokens]
    

    def sort_masks(self, batch_masks):
        """
        Sorts masks in batch_masks by area.

        Args:
            batch_masks (torch.Tensor): Batch of masks.

        Returns:
            torch.Tensor: Sorted batch of masks.
        """
        sums = torch.sum(batch_masks, dim=(2, 3))
        sorted_indices = torch.argsort(sums, dim=1, descending=True)
        for i in range(batch_masks.shape[0]):
            batch_masks[i] = batch_masks[i][sorted_indices[i]]
        return batch_masks