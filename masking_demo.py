import numpy as np
import cv2
from PIL import Image
from functools import wraps
from time import time
from pathlib import Path
from skimage import measure, morphology
from scipy import ndimage
import torch

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

    @timing
    def get_mask(self, img, category='full_body'):
        # Resize image to 512x512 for SegBody
        img_resized = img.resize((512, 512), Image.LANCZOS)
        
        # Use SegBody to create the initial mask
        _, segbody_mask = segment_body(img_resized, face=False)
        segbody_mask = np.array(segbody_mask)
        
        # Use the parsing model to get detailed segmentation
        parse_result, _ = self.parsing_model(img_resized)
        parse_array = np.array(parse_result)
        
        # Create masks for face, head, and hair
        face_head_mask = np.isin(parse_array, [self.label_map["head"], self.label_map["neck"]])
        hair_mask = (parse_array == self.label_map["hair"])
        
        # Combine SegBody mask with face, head, and hair masks
        combined_mask = np.logical_and(segbody_mask > 128, np.logical_not(np.logical_or(face_head_mask, hair_mask)))
        
        # Apply refinement techniques
        refined_mask = self.refine_mask(combined_mask)
        smooth_mask = self.smooth_edges(refined_mask, sigma=1.0)
        expanded_mask = self.expand_mask(smooth_mask)
        
        # Ensure face, head, and hair are not masked
        final_mask = np.logical_and(expanded_mask, np.logical_not(np.logical_or(face_head_mask, hair_mask)))
        
        # Convert to PIL Image
        mask_binary = Image.fromarray((final_mask * 255).astype(np.uint8))
        mask_gray = Image.fromarray((final_mask * 127).astype(np.uint8))
        
        # Resize masks back to original image size
        mask_binary = mask_binary.resize(img.size, Image.LANCZOS)
        mask_gray = mask_gray.resize(img.size, Image.LANCZOS)
        
        return mask_binary, mask_gray

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
    input_folder = Path("/Users/ateeb.taseer/arbi_tryon/arbi-tryon/in_im")
    output_folder = Path("/Users/ateeb.taseer/arbi_tryon/arbi-tryon/output")
    category = "dresses"  # Change to "upper_body", "lower_body", or "dresses" as needed
    
    process_images(str(input_folder), str(output_folder), category)