import numpy as np
import cv2
from PIL import Image
from functools import wraps
from time import time
import mediapipe as mp
from Masking.preprocess.humanparsing.run_parsing import Parsing
from Masking.preprocess.openpose.run_openpose import OpenPose

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

class Masking:
    def __init__(self):
        self.parsing_model = Parsing(-1)
        self.openpose_model = OpenPose(-1)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        self.label_map = {
            "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
            "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
            "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
            "bag": 16, "scarf": 17, "neck": 18
        }

    @timing
    def get_mask(self, img, category='full_body'):
        # Resize image to 384x512 for processing
        img_resized = img.resize((384, 512), Image.LANCZOS)
        img_np = np.array(img_resized)
        
        # Get human parsing result
        parse_result, _ = self.parsing_model(img_resized)
        parse_array = np.array(parse_result)

        # Get pose estimation
        keypoints = self.openpose_model(img_resized)
        pose_data = np.array(keypoints["pose_keypoints_2d"]).reshape((-1, 2))

        # Create initial mask based on category
        if category == 'upper_body':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"]])
        elif category == 'lower_body':
            mask = np.isin(parse_array, [self.label_map["pants"], self.label_map["skirt"]])
        elif category == 'full_body':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"], 
                                         self.label_map["pants"], self.label_map["skirt"]])
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'full_body'.")

        # Enhance mask for white clothing
        white_mask = self.detect_white_clothing(img_np)
        enhanced_mask = np.logical_or(mask, white_mask)

        # Refine the mask
        refined_mask = self.refine_mask(enhanced_mask)

        # Remove hands from the mask
        hand_mask = self.create_hand_mask(img_np)
        refined_mask = np.logical_and(refined_mask, np.logical_not(hand_mask))

        # Apply smoothing and feathering
        final_mask = self.apply_smoothing_and_feathering(refined_mask)

        # Resize mask back to original image size
        mask_pil = Image.fromarray((final_mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(img.size, Image.LANCZOS)
        
        return np.array(mask_pil)

    def detect_white_clothing(self, image):
        # Convert to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab_image[:,:,0]
        
        # Threshold the L channel to identify white areas
        _, white_mask = cv2.threshold(l_channel, 220, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        return white_mask > 0

    def create_hand_mask(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        hand_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_polygon = []
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    hand_polygon.append([x, y])
                
                hand_polygon = np.array(hand_polygon, dtype=np.int32)
                cv2.fillPoly(hand_mask, [hand_polygon], 255)
        
        # Dilate the hand mask to ensure complete coverage
        kernel = np.ones((5,5), np.uint8)
        hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)
        
        return hand_mask > 0

    def refine_mask(self, mask):
        # Convert to uint8 for OpenCV operations
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Apply morphological operations to smooth the mask
        kernel = np.ones((5,5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # Find contours and keep only the largest ones
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]  # Keep top 2 contours
        
        mask_refined = np.zeros_like(mask_uint8)
        cv2.drawContours(mask_refined, contours, -1, 255, -1)
        
        return mask_refined > 0

    def apply_smoothing_and_feathering(self, mask, blur_radius=15, feather_amount=10):
        # Apply Gaussian blur for smoothing
        mask_float = mask.astype(float)
        mask_blurred = cv2.GaussianBlur(mask_float, (blur_radius, blur_radius), 0)
        
        # Create a gradient for feathering
        gradient = np.zeros_like(mask_float)
        gradient = cv2.distanceTransform((1 - mask_float).astype(np.uint8), cv2.DIST_L2, 5)
        gradient = gradient.astype(float) / feather_amount
        gradient = np.clip(gradient, 0, 1)
        
        # Apply feathering
        mask_feathered = mask_blurred * (1 - gradient)
        
        return mask_feathered

    @staticmethod
    def hole_fill(img):
        img_copy = img.copy()
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(img_copy, mask, (0,0), 255)
        filled_image = cv2.bitwise_not(img_copy)
        return cv2.bitwise_or(img, filled_image)

import os
from PIL import Image
from Masking.masking import Masking


if __name__ == "__main__":
    masker = Masking()
    image_folder = "/Users/ikramali/projects/arbiosft_products/arbi-tryon/TEST"
    input_image = os.path.join(image_folder, "mota.jpg")
    output_mask = os.path.join(image_folder, "output_smooth_mask1.png")
    output_masked = os.path.join(image_folder, "output_masked_image1.png")
    category = "dresses"  # Change this to "upper_body", "lower_body", or "dresses" as needed
    
    # Load the input image
    input_img = Image.open(input_image)
    
    # Get the mask
    mask = masker.get_mask(input_img, category=category)
    
    # Save the output mask image
    Image.fromarray(mask).save(output_mask)
    
    # Apply the mask to the input image
    masked_output = input_img.copy()
    masked_output.putalpha(Image.fromarray(mask))
    
    # Save the masked output image
    masked_output.save(output_masked)
    
    print(f"Mask saved to {output_mask}")
    print(f"Masked output saved to {output_masked}")