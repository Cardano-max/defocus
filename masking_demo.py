import numpy as np
import cv2
from PIL import Image, ImageDraw
from functools import wraps
from time import time
import mediapipe as mp

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
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    @timing
    def get_mask(self, img, category='upper_body'):
        # Convert PIL Image to numpy array
        img_np = np.array(img)
        
        # Get pose estimation
        pose_results = self.pose.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        
        # Create initial mask based on category and pose
        mask = self.create_initial_mask(img_np, pose_results, category)
        
        # Refine mask with hand detection
        mask = self.refine_mask_with_hands(img_np, mask)
        
        # Apply smooth transition to edges
        mask = self.apply_smooth_transition(mask)
        
        return mask

    def create_initial_mask(self, image, pose_results, category):
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            if category == 'upper_body':
                self.draw_upper_body_mask(mask, landmarks, height, width)
            elif category == 'lower_body':
                self.draw_lower_body_mask(mask, landmarks, height, width)
            elif category == 'dresses':
                self.draw_full_body_mask(mask, landmarks, height, width)
            else:
                raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")
        
        return mask

    def draw_upper_body_mask(self, mask, landmarks, height, width):
        body_pts = [
            (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height),
            (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * height),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * height),
        ]
        cv2.fillPoly(mask, [np.array(body_pts, dtype=np.int32)], 255)

    def draw_lower_body_mask(self, mask, landmarks, height, width):
        body_pts = [
            (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * height),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * height),
            (landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * height),
        ]
        cv2.fillPoly(mask, [np.array(body_pts, dtype=np.int32)], 255)

    def draw_full_body_mask(self, mask, landmarks, height, width):
        body_pts = [
            (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height),
            (landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * height),
        ]
        cv2.fillPoly(mask, [np.array(body_pts, dtype=np.int32)], 255)

    def refine_mask_with_hands(self, image, mask):
        hand_results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    cv2.circle(mask, (x, y), 15, 0, -1)  # Remove hands from mask
        
        return mask

    def apply_smooth_transition(self, mask, blur_radius=15):
        # Apply Gaussian blur to create smooth transition
        blurred_mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
        
        # Create a gradient transition
        gradient_mask = np.zeros_like(mask)
        cv2.circle(gradient_mask, (mask.shape[1]//2, mask.shape[0]//2), 
                   min(mask.shape[0], mask.shape[1])//2, 255, -1)
        gradient_mask = cv2.GaussianBlur(gradient_mask, (blur_radius*2+1, blur_radius*2+1), 0)
        
        # Combine original mask, blurred mask, and gradient
        final_mask = np.where(gradient_mask > 127, mask, blurred_mask)
        
        return final_mask

if __name__ == "__main__":
    image_folder = "/Users/ikramali/projects/arbiosft_products/arbi-tryon/TEST"
    input_image = os.path.join(image_folder, "mota2.png")  # Replace "input_image.jpg" with your actual image file name
    output_image = os.path.join(image_folder, "output_mask.png")
    category = "dresses"  # Change this to "lower_body" or "dresses" as needed
    
    # Load the input image
    input_img = Image.open(input_image)
    
    # Create an instance of Masking
    masking = Masking()
    
    # Get the mask
    mask = masking.get_mask(input_img, category=category)
    
    # Save the output mask image
    output_img = Image.fromarray(mask)
    output_img.save(output_image)
    
    print(f"Mask saved to {output_image}")
