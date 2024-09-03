import numpy as np
import cv2
from PIL import Image
from functools import wraps
from time import time
from pathlib import Path
from skimage import measure, morphology
from scipy import ndimage
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.info(f'func:{f.__name__} args:[{args}, {kw}] took: {te-ts:.4f} sec')
        return result
    return wrap

class Masking:
    def __init__(self):
        logger.info("Initializing Masking class")
        try:
            self.parsing_model = Parsing(-1)
            self.label_map = {
                "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
                "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
                "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
                "bag": 16, "scarf": 17, "neck": 18
            }
            logger.info("Masking class initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Masking class: {str(e)}")
            raise

    @timing
    def get_mask(self, img, category='full_body'):
        logger.info(f"Getting mask for category: {category}")
        try:
            # Resize image to 512x512 for SegBody
            img_resized = img.resize((512, 512), Image.LANCZOS)
            logger.info("Image resized to 512x512")
            
            # Use SegBody to create the initial mask
            logger.info("Applying SegBody")
            _, segbody_mask = segment_body(img_resized, face=False)
            segbody_mask = np.array(segbody_mask)
            
            # Use the parsing model to get detailed segmentation
            logger.info("Applying parsing model")
            parse_result, _ = self.parsing_model(img_resized)
            parse_array = np.array(parse_result)
            
            # Create masks for face, head, and hair
            logger.info("Creating face, head, and hair masks")
            face_head_mask = np.isin(parse_array, [self.label_map["head"], self.label_map["neck"]])
            hair_mask = (parse_array == self.label_map["hair"])
            
            # Combine SegBody mask with face, head, and hair masks
            logger.info("Combining masks")
            combined_mask = np.logical_and(segbody_mask > 128, np.logical_not(np.logical_or(face_head_mask, hair_mask)))
            
            # Apply refinement techniques
            logger.info("Refining mask")
            refined_mask = self.refine_mask(combined_mask)
            smooth_mask = self.smooth_edges(refined_mask, sigma=1.0)
            expanded_mask = self.expand_mask(smooth_mask)
            
            # Ensure face, head, and hair are not masked
            logger.info("Finalizing mask")
            final_mask = np.logical_and(expanded_mask, np.logical_not(np.logical_or(face_head_mask, hair_mask)))
            
            # Convert to PIL Image
            mask_binary = Image.fromarray((final_mask * 255).astype(np.uint8))
            mask_gray = Image.fromarray((final_mask * 127).astype(np.uint8))
            
            # Resize masks back to original image size
            logger.info("Resizing mask to original image size")
            mask_binary = mask_binary.resize(img.size, Image.LANCZOS)
            mask_gray = mask_gray.resize(img.size, Image.LANCZOS)
            
            logger.info("Mask generation completed successfully")
            return np.array(mask_binary)
        except Exception as e:
            logger.error(f"Error in get_mask: {str(e)}")
            raise

    def refine_mask(self, mask):
        logger.info("Refining mask")
        try:
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
            
            logger.info("Mask refinement completed")
            return mask_refined > 0
        except Exception as e:
            logger.error(f"Error in refine_mask: {str(e)}")
            raise

    def smooth_edges(self, mask, sigma=1.0):
        logger.info("Smoothing mask edges")
        try:
            mask_float = mask.astype(float)
            mask_blurred = ndimage.gaussian_filter(mask_float, sigma=sigma)
            mask_smooth = (mask_blurred > 0.5).astype(np.uint8)
            logger.info("Edge smoothing completed")
            return mask_smooth
        except Exception as e:
            logger.error(f"Error in smooth_edges: {str(e)}")
            raise

    def expand_mask(self, mask, expansion=3):
        logger.info("Expanding mask")
        try:
            kernel = np.ones((expansion, expansion), np.uint8)
            expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            logger.info("Mask expansion completed")
            return expanded_mask > 0
        except Exception as e:
            logger.error(f"Error in expand_mask: {str(e)}")
            raise