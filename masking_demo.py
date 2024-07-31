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
    def get_mask(self, img, category='upper_body'):
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
        mask = self.refine_mask(mask)

        # Apply Gaussian blur to create soft edges
        mask = self.apply_gaussian_blur(mask)

        # Apply edge feathering
        mask = self.apply_edge_feathering(mask)

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
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    cv2.circle(hand_mask, (x, y), 15, 255, -1)
        
        return hand_mask > 0

    def refine_mask(self, mask):
        # Convert to uint8 for OpenCV operations
        mask_uint8 = (mask * 255).astype(np.uint8)
        
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
        
        return mask_refined > 0  # Convert back to boolean mask

    def apply_gaussian_blur(self, mask, kernel_size=15, sigma=5):
        # Apply Gaussian blur to create soft edges
        mask_float = mask.astype(np.float32)
        blurred_mask = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), sigma)
        return blurred_mask

    def apply_edge_feathering(self, mask, feather_amount=10):
        # Create a distance transform of the mask
        dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        
        # Normalize the distance transform
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        
        # Create a feathered mask
        feathered_mask = np.clip(dist_transform * (255.0 / feather_amount), 0, 1)
        
        return feathered_mask

    @staticmethod
    def extend_arm_mask(wrist, elbow, scale):
        wrist = elbow + scale * (wrist - elbow)
        return wrist

    @staticmethod
    def hole_fill(img):
        img = np.pad(img[1:-1, 1:-1], pad_width=1, mode='constant', constant_values=0)
        img_copy = img.copy()
        mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

        cv2.floodFill(img, mask, (0, 0), 255)
        img_inverse = cv2.bitwise_not(img)
        dst = cv2.bitwise_or(img_copy, img_inverse)
        return dst

import os
from PIL import Image
from Masking.masking import Masking

if __name__ == "__main__":
    masker = Masking()
    image_folder = "/Users/ikramali/projects/arbiosft_products/arbi-tryon/TEST"
    input_image = os.path.join(image_folder, "no.png")
    output_mask = os.path.join(image_folder, "output_smooth_mask.png")
    output_masked = os.path.join(image_folder, "output_masked_image.png")
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