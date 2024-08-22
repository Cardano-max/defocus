import numpy as np
from PIL import Image
from pathlib import Path
from functools import wraps
from time import time
import torch
import cv2
import mediapipe as mp

# Ensure torch uses CPU if CUDA is not available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import SegBody (assuming it's in the same directory)
from SegBody import segment_body

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
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    def detect_hands(self, image):
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in hand_landmarks.landmark]
                hull = cv2.convexHull(np.array(points))
                cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def detect_feet(self, image):
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_foot = [(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y),
                         (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value].x,
                          landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value].y),
                         (landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                          landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value].y)]
            right_foot = [(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y),
                          (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value].x,
                           landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value].y),
                          (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                           landmarks[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y)]
            
            for foot in [left_foot, right_foot]:
                points = [(int(x * image.shape[1]), int(y * image.shape[0])) for x, y in foot]
                hull = cv2.convexHull(np.array(points))
                cv2.fillConvexPoly(mask, hull, 255)
        return mask

    @timing
    def get_mask(self, img, category='full_body'):
        # Resize image to 512x512
        img_resized = img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(img_resized)
        
        # Use SegBody to create the mask
        _, mask_image = segment_body(img_resized, face=False)
        
        # Convert mask to binary (0 or 255)
        mask_array = np.array(mask_image)
        binary_mask = (mask_array > 128).astype(np.uint8) * 255
        
        # Detect hands and feet
        hands_mask = self.detect_hands(img_np)
        feet_mask = self.detect_feet(img_np)
        
        # Combine hands and feet masks
        extremities_mask = cv2.bitwise_or(hands_mask, feet_mask)
        
        # Remove hands and feet from the body mask
        body_mask = cv2.bitwise_and(binary_mask, cv2.bitwise_not(extremities_mask))
        
        # Create binary mask PIL Image
        mask_binary = Image.fromarray(body_mask)
        
        # Create grayscale mask
        grayscale_mask = (body_mask > 128).astype(np.uint8) * 127
        mask_gray = Image.fromarray(grayscale_mask)
        
        # Resize masks back to original image size
        mask_binary = mask_binary.resize(img.size, Image.LANCZOS)
        mask_gray = mask_gray.resize(img.size, Image.LANCZOS)
        
        return mask_binary, mask_gray

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
        
        # Save the binary mask
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
    category = "full_body"  # Change to "upper_body", "lower_body", or "dresses" as needed
    
    process_images(str(input_folder), str(output_folder), category)