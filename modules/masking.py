import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from preprocess.openpose.run_openpose import OpenPose
import mediapipe as mp

class Masking:
    def __init__(self):
        self.processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")
        self.openpose = OpenPose(gpu_id=0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)
        self.label_map = {
            "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
            "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
            "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
            "bag": 16, "scarf": 17
        }

    def detect_hands(self, image):
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        hand_masks = np.zeros(image.shape[:2], dtype=np.uint8)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [[int(lm.x * image.shape[1]), int(lm.y * image.shape[0])] for lm in hand_landmarks.landmark]
                hull = cv2.convexHull(np.array(landmarks))
                cv2.fillConvexPoly(hand_masks, hull, 255)
        return hand_masks

    def get_pose_mask(self, pose_data, shape):
        pose_mask = np.zeros(shape, dtype=np.uint8)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Right arm
            (1, 5), (5, 6), (6, 7),  # Left arm
            (1, 8), (8, 9), (9, 10),  # Right leg
            (1, 11), (11, 12), (12, 13)  # Left leg
        ]
        for connection in connections:
            start_point = tuple(map(int, pose_data[connection[0]]))
            end_point = tuple(map(int, pose_data[connection[1]]))
            cv2.line(pose_mask, start_point, end_point, 255, thickness=10)
        return pose_mask

    def get_mask(self, image, category='upper_body', width=384, height=512):
        # Ensure image is in RGB mode and resize
        image = image.convert('RGB').resize((width, height), Image.NEAREST)
        np_image = np.array(image)

        # Get segmentation
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        parse_array = upsampled_logits.argmax(dim=1)[0].numpy()

        # Get pose estimation
        keypoints = self.openpose(np_image)
        pose_data = np.array(keypoints["pose_keypoints_2d"]).reshape((-1, 2))
        pose_mask = self.get_pose_mask(pose_data, (height, width))

        # Detect hands
        hand_mask = self.detect_hands(np_image)

        # Create initial mask based on category
        if category == 'upper_body':
            initial_mask = ((parse_array == 4) | (parse_array == 7)).astype(np.uint8) * 255
        elif category == 'lower_body':
            initial_mask = ((parse_array == 6) | (parse_array == 12) | 
                            (parse_array == 13) | (parse_array == 5)).astype(np.uint8) * 255
        elif category == 'dresses':
            initial_mask = ((parse_array == 7) | (parse_array == 4) | 
                            (parse_array == 5) | (parse_array == 6)).astype(np.uint8) * 255
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

        # Combine initial mask with pose mask
        combined_mask = cv2.bitwise_or(initial_mask, pose_mask)

        # Dilate the combined mask
        kernel = np.ones((5,5), np.uint8)
        dilated_mask = cv2.dilate(combined_mask, kernel, iterations=3)

        # Remove hand areas from the dilated mask
        final_mask = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(hand_mask))

        # Refine the mask
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refined_mask = np.zeros_like(final_mask)
        cv2.drawContours(refined_mask, contours, -1, 255, thickness=cv2.FILLED)

        # Invert the mask for inpainting (white areas will be inpainted)
        inpaint_mask = cv2.bitwise_not(refined_mask)

        return Image.fromarray(inpaint_mask)