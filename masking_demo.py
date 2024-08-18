import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from functools import wraps
from time import time
from Masking.preprocess.humanparsing.run_parsing import Parsing
from Masking.preprocess.openpose.run_openpose import OpenPose
from pathlib import Path
from skimage import measure
from scipy import ndimage
from rembg import remove

import numpy as np
import cv2
from PIL import Image
import mediapipe as mp


class Masking:
    def __init__(self):
        self.parsing_model = Parsing(-1)
        self.openpose_model = OpenPose(-1)
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.7)
        self.label_map = {
            "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
            "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
            "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
            "bag": 16, "scarf": 17, "neck": 18
        }

    def get_mask(self, img, category='upper_body'):
        # Resize image to 512x512 for processing
        img_resized = img.resize((512, 512), Image.LANCZOS)
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

        # Create body parts mask
        body_mask = np.isin(parse_array, [self.label_map["left_arm"], self.label_map["right_arm"],
                                          self.label_map["left_leg"], self.label_map["right_leg"],
                                          self.label_map["head"], self.label_map["hair"],
                                          self.label_map["neck"]])

        # Create hand mask using MediaPipe
        hand_mask = self.create_hand_mask(img_np)

        # Create face and hair mask using MediaPipe Face Mesh
        face_hair_mask = self.create_face_hair_mask(img_np)

        # Combine all body part masks
        body_parts_mask = np.logical_or(np.logical_or(body_mask, hand_mask), face_hair_mask)

        # Remove body parts from the garment mask
        mask = np.logical_and(mask, np.logical_not(body_parts_mask))

        # Refine the mask
        mask = self.refine_mask(mask)

        # Ensure the mask covers the entire garment
        mask = self.expand_mask(mask)

        # Resize mask back to original image size
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize(img.size, Image.LANCZOS)
        
        # Create garment mask
        garment_mask = img.copy()
        garment_mask.putalpha(mask_pil)
        
        return mask_pil, garment_mask

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

    def create_face_hair_mask(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        face_hair_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_polygon = []
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    face_polygon.append([x, y])
                
                face_polygon = np.array(face_polygon, dtype=np.int32)
                cv2.fillPoly(face_hair_mask, [face_polygon], 255)
        
        # Expand the face mask to include potential hair regions
        kernel = np.ones((20,20), np.uint8)
        face_hair_mask = cv2.dilate(face_hair_mask, kernel, iterations=2)
        
        return face_hair_mask > 0

    def refine_mask(self, mask):
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
        
        return mask_refined > 0

    def expand_mask(self, mask):
        # Dilate the mask to ensure complete coverage of the garment
        kernel = np.ones((10,10), np.uint8)
        expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=3)
        
        # Fill any holes in the mask
        expanded_mask = self.hole_fill(expanded_mask)
        
        return expanded_mask > 0

    @staticmethod
    def hole_fill(img):
        img = np.pad(img, pad_width=1, mode='constant', constant_values=0)
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(img, mask, (0,0), 255)
        img = cv2.bitwise_not(img)
        return img[1:-1, 1:-1]

def process_images(input_folder, output_folder, category, output_format='png'):
    masker = Masking()
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all image files from the input folder
    image_files = list(Path(input_folder).glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')]
    
    for i, image_file in enumerate(image_files, 1):
        output_mask = Path(output_folder) / f"output_sharp_mask_{i}.{output_format}"
        output_masked = Path(output_folder) / f"output_masked_image_white_bg_{i}.{output_format}"
        output_upscaled = Path(output_folder) / f"output_upscaled_8k_{i}.{output_format}"
        
        print(f"Processing image {i}/{len(image_files)}: {image_file.name}")
        
        input_img = Image.open(image_file).convert('RGB')
        
        mask = masker.get_mask(input_img, category=category)
        
        Image.fromarray(mask).save(str(output_mask))
        
        # Create a white background image
        white_bg = Image.new('RGB', input_img.size, (255, 255, 255))
        
        # Create a new image with white background and paste the masked input image
        masked_output = Image.composite(input_img, white_bg, Image.fromarray(mask))
        
        # Remove background using rembg for better results
        masked_output_removed_bg = remove(np.array(masked_output))
        masked_output_removed_bg = Image.fromarray(masked_output_removed_bg)
        
        if output_format.lower() == 'webp':
            masked_output_removed_bg.save(str(output_masked), format='WebP', lossless=True)
        else:
            masked_output_removed_bg.save(str(output_masked))
        
        # Upscale to 8K resolution
        target_size = (7680, 4320)  # 8K resolution
        upscaled_output = masked_output_removed_bg.resize(target_size, Image.LANCZOS)
        
        if output_format.lower() == 'webp':
            upscaled_output.save(str(output_upscaled), format='WebP', lossless=True)
        else:
            upscaled_output.save(str(output_upscaled))
        
        print(f"Mask saved to {output_mask}")
        print(f"Masked output with white background saved to {output_masked}")
        print(f"Upscaled 8K output saved to {output_upscaled}")
        print()

if __name__ == "__main__":
    input_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/Input_Images")
    output_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/output")
    category = "dresses"  # Change this to "upper_body", "lower_body", or "dresses" as needed
    output_format = "webp"  # Change this to "png" if you prefer PNG output
    
    process_images(str(input_folder), str(output_folder), category, output_format)