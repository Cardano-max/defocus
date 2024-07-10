import cv2
import torch
import numpy as np
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")

def mask_clothes(image, labels=[4,14,15]):
    # image = Image.open("/content/images.jpeg")
    inputs = processor(images=image, return_tensors="pt")


    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    # Upper Clothes 4, Left Arm 14, Right Arm 15
    # Create masks where the values are 4 and 6
    # Use torch.logical_or to combine the masks
    if len(labels) > 1:
        combined_mask = torch.logical_or(pred_seg == labels[0], pred_seg == labels[1])
    else:
        combined_mask = pred_seg == labels[0]
    if len(labels) > 2:
        for i in range(2, len(labels)):
            combined_mask = torch.logical_or(combined_mask, pred_seg == labels[i])

    # Create a new tensor filled with 0s 
    pred_seg_new = torch.zeros_like(pred_seg)

    # Set the values at the positions where the mask is True to 255 
    pred_seg_new[combined_mask] = 255

    image_mask = pred_seg_new.numpy().astype(np.uint8)

    # Define the kernel size for dilation. The larger the kernel, the more the mask will be dilated.
    kernel_size = 50
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply dilation to the mask
    dilated_mask = cv2.dilate(image_mask, kernel, iterations=1)

    return dilated_mask
