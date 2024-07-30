# Masking/masking.py

import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from functools import wraps
from time import time

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
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

    @timing
    def get_mask(self, img, category='upper_body'):
        # Convert PIL Image to numpy array
        img_np = np.array(img)
        
        # Get hand mask
        hand_mask = self.get_hand_mask(img_np)
        
        # Get body mask based on category
        body_mask = self.get_body_mask(img_np, category)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(hand_mask, body_mask)
        
        # Refine the mask
        refined_mask = self.refine_mask(combined_mask)
        
        return refined_mask

    def get_hand_mask(self, image):
        height, width, _ = image.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    cv2.circle(mask, (x, y), 15, 255, -1)
        
        # Dilate the mask to create a more generous hand area
        kernel = np.ones((20, 20), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask

    def get_body_mask(self, image, category):
        height, width, _ = image.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            if category == 'upper_body':
                body_parts = [11, 12, 23, 24]  # Shoulders and hips
            elif category == 'lower_body':
                body_parts = [23, 24, 25, 26, 27, 28]  # Hips, knees, and ankles
            elif category == 'dresses':
                body_parts = [11, 12, 23, 24, 25, 26, 27, 28]  # Full body
            else:
                raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")
            
            points = []
            for i in body_parts:
                x, y = int(landmarks[i].x * width), int(landmarks[i].y * height)
                points.append((x, y))
            
            cv2.fillPoly(mask, [np.array(points)], 255)
        
        return mask

    def refine_mask(self, mask):
        # Apply morphological operations to smooth the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours and keep only the largest one
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_refined = np.zeros_like(mask)
            cv2.drawContours(mask_refined, [largest_contour], 0, 255, -1)
        else:
            mask_refined = mask
        
        return mask_refined