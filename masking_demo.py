import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from Masking.preprocess.humanparsing.run_parsing import Parsing
from Masking.preprocess.openpose.run_openpose import OpenPose
import os
from tqdm import tqdm

class ImprovedMasking:
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
            mask = np.isin(parse_array, [self.label_map["upper_clothes"]])
        elif category == 'lower_body':
            mask = np.isin(parse_array, [self.label_map["pants"], self.label_map["skirt"]])
        elif category == 'dresses':
            mask = np.isin(parse_array, [self.label_map["dress"]])
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

        # Create body parts mask
        body_parts_mask = np.isin(parse_array, [
            self.label_map["left_arm"], self.label_map["right_arm"],
            self.label_map["left_leg"], self.label_map["right_leg"],
            self.label_map["head"], self.label_map["neck"],
            self.label_map["hair"]
        ])

        # Create hand mask using MediaPipe
        hand_mask = self.create_hand_mask(img_np)

        # Create face mask using MediaPipe Face Mesh
        face_mask = self.create_face_mask(img_np)

        # Combine all body part masks
        combined_body_mask = np.logical_or(np.logical_or(body_parts_mask, hand_mask), face_mask)

        # Remove body parts from the garment mask
        mask = np.logical_and(mask, np.logical_not(combined_body_mask))

        # Refine the mask
        mask = self.refine_mask(mask)

        # Fill holes in the mask
        mask = self.hole_fill(mask.astype(np.uint8) * 255)

        # Resize mask back to original image size
        mask_pil = Image.fromarray(mask)
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
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    hand_points.append([x, y])
                hand_points = np.array(hand_points, dtype=np.int32)
                cv2.fillPoly(hand_mask, [hand_points], 255)
        
        # Dilate the hand mask to ensure complete coverage
        kernel = np.ones((5,5), np.uint8)
        hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)
        
        return hand_mask > 0

    def create_face_mask(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_points = []
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    face_points.append([x, y])
                face_points = np.array(face_points, dtype=np.int32)
                cv2.fillPoly(face_mask, [face_points], 255)
        
        # Dilate the face mask to ensure complete coverage
        kernel = np.ones((5,5), np.uint8)
        face_mask = cv2.dilate(face_mask, kernel, iterations=2)
        
        return face_mask > 0

    def refine_mask(self, mask):
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

    @staticmethod
    def hole_fill(img):
        img = np.pad(img, pad_width=1, mode='constant', constant_values=0)
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(img, mask, (0,0), 255)
        img = cv2.bitwise_not(img)
        return img[1:-1, 1:-1]

def process_images(input_folder, output_folder, masking_instance):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    for image_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f"masked_{image_file}")
        
        img = Image.open(input_path).convert("RGB")
        
        # Determine the category based on the image or use a default
        # You may need to implement a method to determine the category
        category = 'upper_body'  # or 'lower_body' or 'dresses'
        
        _, garment_mask = masking_instance.get_mask(img, category)
        garment_mask.save(output_path)

if __name__ == "__main__":
    input_folder = "/Users/ikramali/projects/arbiosft_products/arbi-tryon/in_im"
    output_folder = "/Users/ikramali/projects/arbiosft_products/arbi-tryon/output"
    
    masking = ImprovedMasking()
    process_images(input_folder, output_folder, masking)