import cv2
import numpy as np
from PIL import Image as PILImage

import torch
import torchvision.transforms as transforms
import torch.distributed as dist

from datasets import Dataset, load_dataset
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, TrainingArguments, Trainer


def annotation_to_label(label_map, line_thickness=3):
    """
    Parameters:
    label_map (PIL.Image): The input label map.
    line_thickness (int): The thickness of the lines that will be drawn for the contours.

    Returns:
    PIL.Image: The output binary boundary label image.
    """
    label_map = np.array(label_map)
    all_contours = []
    for label_idx in np.unique(label_map):
        mask = (label_map == label_idx).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.append(contours)
    h, w = label_map.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for contours in all_contours:
        cv2.drawContours(canvas, contours, -1, (1, 1, 1), line_thickness)
    label = PILImage.fromarray(canvas[:, :, 0], mode='L')
    return label


def transforms(example_batch):
    images = [x.convert("RGB") for x in example_batch["image"]]
    labels = [annotation_to_label(x) for x in example_batch["annotation"]]
    inputs = image_processor(images, labels, do_reduce_labels=False)
    return inputs


if __name__=='__main__':
    
    dist.init_process_group(backend='nccl')

    dataset = load_dataset("scene_parse_150", split="train")
    dataset.set_transform(transforms)

    checkpoint = "chendelong/DirectSAM-1800px-0424"
    model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, num_labels=1, ignore_mismatched_sizes=True)
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=True)

    input_resolution = 512
    image_processor.size['height'] = input_resolution
    image_processor.size['width'] = input_resolution
    
    if torch.distributed.get_rank() == 0:
        print(model)
        print(f"Number of parameters: {model.num_parameters()/1e6}M,  trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6}M")
        print(dataset)

    training_args = TrainingArguments(
        output_dir=f'runs/finetune-directsam-ade20k-5ep-512px',
        learning_rate=5e-5,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        save_total_limit=3,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=4,
        save_strategy="epoch",
        do_eval=False,
        logging_steps=1,
        remove_unused_columns=False,
        push_to_hub=False,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

