import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor, Mask2FormerForUniversalSegmentation, OneFormerForUniversalSegmentation, OneFormerProcessor

class PanopticSegmenter:

    def __init__(
            self, 
            checkpoint="facebook/mask2former-swin-tiny-coco-panoptic", 
            image_resolution=1024, 
            max_tokens=128, 
            device="cuda", 
            **kwargs
            ):
        self.checkpoint = checkpoint
        self.image_resolution = image_resolution
        self.max_tokens = max_tokens
        self.device = device

        if 'mask2former' in checkpoint.lower():
            self.image_processor = AutoImageProcessor.from_pretrained(checkpoint)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint).to(self.device).half().eval()
        elif 'oneformer' in checkpoint.lower():
            self.image_processor = OneFormerProcessor.from_pretrained(checkpoint)
            self.model = OneFormerForUniversalSegmentation.from_pretrained(checkpoint).to(self.device).half().eval()
        else:
            raise NotImplementedError(f"Panoptic mode '{checkpoint}' is not supported")

    def load_images(self, images):
        if not isinstance(images, list):
            images = [images]
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.convert("RGB").resize((self.image_resolution, self.image_resolution))
            processed_images.append(img)
        return processed_images

    @torch.inference_mode()
    def __call__(self, images):
        images = self.load_images(images)
        batch_masks = np.zeros(
            (len(images), self.max_tokens, self.image_resolution, self.image_resolution),
            dtype=bool
        )
        if 'mask2former' in self.checkpoint.lower():
            batch_masks = self._process_mask2former(images, batch_masks)
        elif 'oneformer' in self.checkpoint.lower():
            batch_masks = self._process_oneformer(images, batch_masks)

        return torch.tensor(batch_masks)

    def _process_mask2former(self, images, batch_masks):
        inputs = self.image_processor(images, return_tensors="pt", task_inputs=["panoptic"])
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device).to(self.model.dtype)
        outputs = self.model(**inputs)
        batch_result = self.image_processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[(self.image_resolution, self.image_resolution) for _ in images], label_ids_to_fuse=set()
        )
        batch_panoptic_map = [result["segmentation"].cpu().numpy() for result in batch_result]
        return self._generate_masks_from_panoptic_map(batch_panoptic_map, batch_masks)

    def _process_oneformer(self, images, batch_masks):
        batch_panoptic_map = []
        for image in images:
            inputs = self.image_processor(image, return_tensors="pt", task_inputs=["panoptic"], size=(self.image_resolution, self.image_resolution))
            inputs["pixel_values"] = inputs["pixel_values"].to(self.device).to(self.model.dtype)
            inputs["task_inputs"] = inputs["task_inputs"].to(self.device).to(self.model.dtype)
            output = self.model(**inputs)
            result = self.image_processor.post_process_panoptic_segmentation(
                output, target_sizes=[(self.image_resolution, self.image_resolution)], label_ids_to_fuse=set())[0]
            batch_panoptic_map.append(result["segmentation"].cpu().numpy())
        return self._generate_masks_from_panoptic_map(batch_panoptic_map, batch_masks)

    def _generate_masks_from_panoptic_map(self, batch_panoptic_map, batch_masks):
        for i, panoptic_map in enumerate(batch_panoptic_map):
            unique_labels = np.unique(panoptic_map)
            masks = np.array([panoptic_map == label for label in unique_labels[:self.max_tokens]])
            batch_masks[i, :len(masks)] = masks
        return batch_masks