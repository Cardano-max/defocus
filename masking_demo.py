# Masking/masking.py

import numpy as np
import cv2
from PIL import Image
from functools import wraps
from time import time
import mediapipe as mp
from Masking.preprocess.humanparsing.run_parsing import Parsing
from Masking.preprocess.openpose.run_openpose import OpenPose
from scipy.ndimage import gaussian_filter

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} took: {te-ts:.4f} sec')
        return result
    return wrap

class Masking:
    def __init__(self):
        self.parsing_model = Parsing(-1)
        self.openpose_model = OpenPose(-1)
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.7)
        self.label_map = {
            "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
            "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
            "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
            "bag": 16, "scarf": 17, "neck": 18
        }

    @timing
    def get_mask(self, img, category='upper_body'):
        # Resize image to 512x512 for processing
        img_resized = img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(img_resized)
        
        # Get human parsing result
        parse_result, _ = self.parsing_model(img_resized)
        parse_array = np.array(parse_result)

        # Get pose estimation
        keypoints = self.openpose_model(img_resized)
        pose_data = np.array(keypoints["pose_keypoints_2d"]).reshape((-1, 3))

        # Create initial mask based on category
        if category == 'upper_body':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"]])
        elif category == 'lower_body':
            mask = np.isin(parse_array, [self.label_map["pants"], self.label_map["skirt"]])
        elif category == 'dresses':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"], 
                                         self.label_map["pants"], self.label_map["skirt"]])
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

        # Create arm mask
        arm_mask = np.isin(parse_array, [self.label_map["left_arm"], self.label_map["right_arm"]])

        # Create hand mask using MediaPipe
        hand_mask = self.create_hand_mask(img_np)

        # Combine arm and hand mask
        arm_hand_mask = np.logical_or(arm_mask, hand_mask)

        # Remove arms and hands from the mask
        mask = np.logical_and(mask, np.logical_not(arm_hand_mask))

        # Refine the mask
        mask = self.refine_mask(mask, img_np)

        # Apply Gaussian blur for smooth transitions
        mask = self.apply_gaussian_blur(mask)

        # Resize mask back to original image size
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(img.size, Image.LANCZOS)
        
        return np.array(mask_pil)

    def create_hand_mask(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        hand_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    hand_points.append([x, y])
                hand_points = np.array(hand_points, dtype=np.int32)
                cv2.fillPoly(hand_mask, [hand_points], 255)
        
        return hand_mask > 0

    def refine_mask(self, mask, image):
        # Convert to uint8 for OpenCV operations
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Apply morphological operations to smooth the mask
        kernel = np.ones((5,5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # Find contours and keep only the largest one
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_refined = np.zeros_like(mask_uint8)
            cv2.drawContours(mask_refined, [largest_contour], 0, 255, -1)
        else:
            mask_refined = mask_uint8
        
        # Use color thresholding to improve detection of white clothes
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine the refined mask with the white mask
        mask_combined = cv2.bitwise_or(mask_refined, white_mask)
        
        return mask_combined > 0

    def apply_gaussian_blur(self, mask, sigma=3):
        # Apply Gaussian blur to create smooth transitions
        blurred_mask = gaussian_filter(mask.astype(float), sigma=sigma)
        
        # Normalize the blurred mask
        blurred_mask = (blurred_mask - blurred_mask.min()) / (blurred_mask.max() - blurred_mask.min())
        
        return blurred_mask

    @staticmethod
    def hole_fill(img):
        img_copy = img.copy()
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(img_copy, mask, (0,0), 255)
        img_floodfill_inv = cv2.bitwise_not(img_copy)
        img_out = img | img_floodfill_inv
        return img_out

import os
from PIL import Image
from Masking.masking import Masking


if __name__ == "__main__":
    masker = Masking()
    image_folder = "/Users/ikramali/projects/arbiosft_products/arbi-tryon/TEST"
    input_image = os.path.join(image_folder, "hania.jpg")
    output_mask = os.path.join(image_folder, "output_smooth_mask2.png")
    output_masked = os.path.join(image_folder, "output_masked_image2.png")
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