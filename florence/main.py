import os
import sys
import numpy as np
import torch
import supervision as sv
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from hydra import compose, initialize
from transformers import AutoTokenizer, AutoImageProcessor

# Add the Florence model directory to the Python path
florence_model_dir = './Florence-2-large-ft'
sys.path.insert(0, florence_model_dir)

# Import the custom Florence modules
from configuration_florence2 import Florence2Config
from modeling_florence2 import Florence2ForConditionalGeneration
from processing_florence2 import Florence2Processor

# Add the SAM model directory to the Python path
sys.path.append('./sam2')

# Import SAM modules
from modeling.sam2_base import SAM2Base
from sam2_image_predictor import SAM2ImagePredictor

# Set up device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Florence model setup
FLORENCE_MODEL_PATH = "./Florence-2-large-ft"

def load_florence_model(device: torch.device, model_path: str) -> tuple:
    config = Florence2Config.from_pretrained(model_path)
    model = Florence2ForConditionalGeneration.from_pretrained(model_path, config=config).to(device).eval()
    return model, config

# SAM model setup
SAM_CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
SAM_CONFIG = "./checkpoints"

def load_sam_image_model(device: torch.device, config_path: str, checkpoint: str = SAM_CHECKPOINT) -> SAM2ImagePredictor:
    with initialize(config_path=config_path, job_name="sam2"):
        cfg = compose(config_name="sam2_hiera_l")
        model = SAM2Base(cfg.model)
        model.to(device)
        ckpt = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=True)

        return SAM2ImagePredictor(sam_model=model)

def run_sam_inference(model: Any, image: np.ndarray, detections: sv.Detections) -> sv.Detections:
    model.set_image(image)
    bboxes = sorted(detections.xyxy, key=lambda bbox: bbox[0])
    mask, score, _ = model.predict(box=bboxes, multimask_output=False)
    if len(mask.shape) == 4:
        mask = np.squeeze(mask)
    detections.mask = mask.astype(bool)
    return detections

# Main processing function
def process_image(
    image_input: str,
    task_prompt: str,
    text_prompt: str = None,
    dilate: int = 0,
    merge_masks: bool = False,
    return_rectangles: bool = False,
    invert_mask: bool = False
) -> list:
    # Load models
    florence_model, florence_config = load_florence_model(DEVICE, FLORENCE_MODEL_PATH)
    sam_model = load_sam_image_model(DEVICE, config_path=SAM_CONFIG)

    # Load tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Florence-2-large")
    image_processor = AutoImageProcessor.from_pretrained("microsoft/Florence-2-large")

    # Create processor instance with tokenizer
    processor = Florence2Processor(image_processor=image_processor, tokenizer=tokenizer)

    # Prepare image
    image = Image.open(image_input)

    # Run Florence inference
    inputs = processor(text=[task_prompt, text_prompt], images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        output = florence_model.generate(**inputs, max_length=512)
    result = processor.post_process_generation(output[0], task=task_prompt, image_size=image.size)

    # Create detections
    detections = sv.Detections.from_lmm(
        lmm=sv.LMM.FLORENCE_2,
        result=result,
        resolution_wh=image.size
    )

    images = []
    if return_rectangles:
        # Generate rectangle masks
        image_width, image_height = image.size
        merge_mask_image = np.zeros((image_height, image_width), dtype=np.uint8)
        bboxes = sorted(detections.xyxy, key=lambda bbox: bbox[0])
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(merge_mask_image, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)
            clip_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            cv2.rectangle(clip_mask, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)
            images.append(clip_mask)
        if merge_masks:
            images = [merge_mask_image] + images
    else:
        # Generate segmentation masks using SAM
        detections = run_sam_inference(sam_model, np.array(image.convert("RGB")), detections)
        if len(detections) == 0:
            print("No objects detected.")
            return None
        print("Masks generated:", len(detections.mask))
        kernel = np.ones((dilate, dilate), np.uint8)

        for i in range(len(detections.mask)):
            mask = detections.mask[i].astype(np.uint8) * 255
            if dilate > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            images.append(mask)

        if merge_masks:
            merged_mask = np.zeros_like(images[0], dtype=np.uint8)
            for mask in images:
                merged_mask = cv2.bitwise_or(merged_mask, mask)
            images = [merged_mask]

    if invert_mask:
        images = [cv2.bitwise_not(mask) for mask in images]

    return images

# Example usage
image_path = "./Jimin.jpeg"
task_prompt = "<OPEN_VOCABULARY_DETECTION>"
text_prompt = "person"

result_masks = process_image(
    image_input=image_path,
    task_prompt=task_prompt,
    text_prompt=text_prompt,
    dilate=10,
    merge_masks=False,
    return_rectangles=False,
    invert_mask=False
)

# Display results
if result_masks:
    plt.figure(figsize=(15, 5))
    for i, mask in enumerate(result_masks):
        plt.subplot(1, len(result_masks), i+1)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Overlay mask on original image
    original_image = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    for mask in result_masks:
        plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.title("Masks overlaid on original image")
    plt.show()
else:
    print("No masks were generated.")