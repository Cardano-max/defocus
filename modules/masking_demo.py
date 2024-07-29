import os
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import mediapipe as mp
from Masking.preprocess.openpose.run_openpose import OpenPose


class Masking:
    def __init__(self):
        self.processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
        self.label_map = {
            0: "background", 1: "hat", 2: "hair", 3: "sunglasses", 4: "upper_clothes",
            5: "skirt", 6: "pants", 7: "dress", 8: "belt", 9: "left_shoe", 10: "right_shoe",
            11: "face", 12: "left_leg", 13: "right_leg", 14: "left_arm", 15: "right_arm",
            16: "bag", 17: "scarf"
        }
        self.color_map = {
            0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (128, 128, 0), 4: (0, 0, 128),
            5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (64, 0, 0), 9: (192, 0, 0),
            10: (64, 128, 0), 11: (192, 128, 0), 12: (64, 0, 128), 13: (192, 0, 128),
            14: (64, 128, 128), 15: (192, 128, 128), 16: (0, 64, 0), 17: (128, 64, 0)
        }

    def detect_hands_and_pose(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(image_rgb)
        pose_results = self.pose.process(image_rgb)
        
        hand_landmarks = []
        if hand_results.multi_hand_landmarks:
            for hand_lms in hand_results.multi_hand_landmarks:
                landmarks = [[lm.x, lm.y] for lm in hand_lms.landmark]
                hand_landmarks.append(landmarks)
        
        pose_landmarks = None
        if pose_results.pose_landmarks:
            pose_landmarks = [[lm.x, lm.y] for lm in pose_results.pose_landmarks.landmark]
        
        return hand_landmarks, pose_landmarks

    def create_hand_mask(self, image_shape, hand_landmarks):
        hand_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        for landmarks in hand_landmarks:
            points = np.array(landmarks) * [image_shape[1], image_shape[0]]
            points = points.astype(int)
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(hand_mask, hull, 255)
        return hand_mask

    def refine_mask(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refined_mask = np.zeros_like(mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Adjust this threshold as needed
                cv2.drawContours(refined_mask, [contour], 0, 255, -1)
        return refined_mask

    def get_mask(self, image_path, category='upper_body'):
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        image_np = np.array(image)
        
        # Segment the image
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        seg_map = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # Create segmentation visualization
        seg_image = np.zeros((height, width, 3), dtype=np.uint8)
        for label, color in self.color_map.items():
            seg_image[seg_map == label] = color
        
        # Detect hands and pose
        hand_landmarks, pose_landmarks = self.detect_hands_and_pose(image_np)
        
        # Create initial mask
        if category == 'upper_body':
            mask = np.isin(seg_map, [4, 7, 14, 15]).astype(np.uint8) * 255
        elif category == 'lower_body':
            mask = np.isin(seg_map, [5, 6, 12, 13]).astype(np.uint8) * 255
        elif category == 'dresses':
            mask = np.isin(seg_map, [4, 5, 6, 7, 14, 15]).astype(np.uint8) * 255
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")
        
        # Refine the mask
        mask = self.refine_mask(mask)
        
        # Create hand mask and unmask hands
        hand_mask = self.create_hand_mask(image_np.shape, hand_landmarks)
        mask[hand_mask > 0] = 0
        
        # Dilate the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Unmask hands again after dilation
        mask[hand_mask > 0] = 0
        
        return seg_image, mask

def main():
    masker = Masking()
    image_path = "arbi-tryon/images/ok.png"  # Replace with your image path
    category = "upper_body"  # or "lower_body" or "dresses"
    
    seg_image, mask = masker.get_mask(image_path, category)
    
    # Save and display results
    cv2.imwrite("segmentation.png", cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite("mask.png", mask)
    
    cv2.imshow("Segmentation", cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()