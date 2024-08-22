import numpy as np
import cv2
from PIL import Image
from functools import wraps
from time import time
from pathlib import Path
from skimage import measure, morphology
from scipy import ndimage
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ensure torch uses CPU if CUDA is not available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import SegBody and other necessary models
from SegBody import segment_body
from Masking.preprocess.humanparsing.run_parsing import Parsing

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} args:[{args}, {kw}] took: {te-ts:.4f} sec')
        return result
    return wrap

class Masking:
    def __init__(self):
        self.parsing_model = Parsing(-1)
        self.label_map = {
            "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
            "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
            "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
            "bag": 16, "scarf": 17, "neck": 18
        }
        
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

    @timing
    def get_mask(self, img, category='full_body'):
        # Resize image to 512x512 for SegBody
        img_resized = img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(img_resized)
        
        # Use SegBody to create the initial mask
        _, segbody_mask = segment_body(img_resized, face=False)
        segbody_mask = np.array(segbody_mask)
        
        # Use the parsing model to get detailed segmentation
        parse_result, _ = self.parsing_model(img_resized)
        parse_array = np.array(parse_result)
        
        # Create masks for face, head, hair, feet, arms
        face_head_mask = np.isin(parse_array, [self.label_map["head"], self.label_map["neck"]])
        hair_mask = (parse_array == self.label_map["hair"])
        feet_mask = np.isin(parse_array, [self.label_map["left_shoe"], self.label_map["right_shoe"]])
        arm_mask = np.isin(parse_array, [self.label_map["left_arm"], self.label_map["right_arm"]])
        
        # Create hand mask
        hand_mask = self.create_precise_hand_mask(img_np)
        
        # Combine all masks that should not be masked
        unmasked_areas = np.logical_or.reduce((face_head_mask, hair_mask, feet_mask, arm_mask, hand_mask))
        
        # Combine SegBody mask with unmasked areas
        combined_mask = np.logical_and(segbody_mask > 128, np.logical_not(unmasked_areas))
        
        # Apply refinement techniques
        refined_mask = self.refine_mask(combined_mask)
        smooth_mask = self.smooth_edges(refined_mask, sigma=1.0)
        expanded_mask = self.expand_mask(smooth_mask)
        
        # Ensure unmasked areas remain unmasked
        final_mask = np.logical_and(expanded_mask, np.logical_not(unmasked_areas))
        
        # Convert to PIL Image
        mask_binary = Image.fromarray((final_mask * 255).astype(np.uint8))
        mask_gray = Image.fromarray((final_mask * 127).astype(np.uint8))
        
        # Resize masks back to original image size
        mask_binary = mask_binary.resize(img.size, Image.LANCZOS)
        mask_gray = mask_gray.resize(img.size, Image.LANCZOS)
        
        return mask_binary, mask_gray

    def create_precise_hand_mask(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        detection_result = self.hand_landmarker.detect(mp_image)
        
        hand_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                hand_points = []
                for landmark in hand_landmarks:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    hand_points.append([x, y])
                hand_points = np.array(hand_points, dtype=np.int32)
                cv2.fillPoly(hand_mask, [hand_points], 1)
        
        # Dilate the hand mask slightly to ensure full coverage
        kernel = np.ones((5,5), np.uint8)
        hand_mask = cv2.dilate(hand_mask, kernel, iterations=1)
        
        return hand_mask > 0

    def refine_mask(self, mask):
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Apply minimal morphological operations
        kernel = np.ones((3,3), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours and keep only the largest one
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_refined = np.zeros_like(mask_uint8)
            cv2.drawContours(mask_refined, [largest_contour], 0, 255, 2)  # Slightly thicker outline
            mask_refined = cv2.fillPoly(mask_refined, [largest_contour], 255)
        else:
            mask_refined = mask_uint8
        
        return mask_refined > 0

    def smooth_edges(self, mask, sigma=1.0):
        mask_float = mask.astype(float)
        mask_blurred = ndimage.gaussian_filter(mask_float, sigma=sigma)
        mask_smooth = (mask_blurred > 0.5).astype(np.uint8)
        return mask_smooth

    def expand_mask(self, mask, expansion=3):
        kernel = np.ones((expansion, expansion), np.uint8)
        expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return expanded_mask > 0

def process_images(input_folder, output_folder, category):
    masker = Masking()
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    image_files = list(Path(input_folder).glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')]
    
    for i, image_file in enumerate(image_files, 1):
        output_mask = Path(output_folder) / f"output_mask_{i}.png"
        output_masked = Path(output_folder) / f"output_masked_image_{i}.png"
        
        print(f"Processing image {i}/{len(image_files)}: {image_file.name}")
        
        input_img = Image.open(image_file).convert('RGB')
        
        mask_binary, mask_gray = masker.get_mask(input_img, category=category)
        
        mask_binary.save(str(output_mask))
        
        # Apply the mask to the original image
        masked_output = Image.composite(input_img, Image.new('RGB', input_img.size, (255, 255, 255)), mask_binary)
        masked_output.save(str(output_masked))
        
        print(f"Mask saved to {output_mask}")
        print(f"Masked output saved to {output_masked}")
        print()

if __name__ == "__main__":
    input_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/in_im")
    output_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/output")
    category = "dresses"  # Change to "upper_body", "lower_body", or "dresses" as needed
    
    process_images(str(input_folder), str(output_folder), category)