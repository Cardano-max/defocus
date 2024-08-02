import numpy as np
import cv2
from PIL import Image
from functools import wraps
from time import time
import mediapipe as mp
from Masking.preprocess.humanparsing.run_parsing import Parsing
from Masking.preprocess.openpose.run_openpose import OpenPose
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay
from skimage.transform import PiecewiseAffineTransform, warp

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
        img_resized = img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(img_resized)
        
        parse_result, _ = self.parsing_model(img_resized)
        parse_array = np.array(parse_result)

        if category == 'upper_body':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"]])
        elif category == 'lower_body':
            mask = np.isin(parse_array, [self.label_map["pants"], self.label_map["skirt"]])
        elif category == 'dresses':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"], 
                                         self.label_map["pants"], self.label_map["skirt"]])
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

        mask = self.refine_mask(mask, img_np)
        mask = self.apply_gaussian_blur(mask)

        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(img.size, Image.LANCZOS)
        
        return np.array(mask_pil)

    def refine_mask(self, mask, image):
        mask_uint8 = mask.astype(np.uint8) * 255
        
        kernel = np.ones((5,5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_refined = np.zeros_like(mask_uint8)
            cv2.drawContours(mask_refined, [largest_contour], 0, 255, -1)
        else:
            mask_refined = mask_uint8
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        mask_combined = cv2.bitwise_or(mask_refined, white_mask)
        
        return mask_combined > 0

    def apply_gaussian_blur(self, mask, sigma=3):
        blurred_mask = gaussian_filter(mask.astype(float), sigma=sigma)
        blurred_mask = (blurred_mask - blurred_mask.min()) / (blurred_mask.max() - blurred_mask.min())
        return blurred_mask

    def generate_landmarks(self, mask, num_points=100):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        landmarks = []
        for point in approx:
            landmarks.append(point[0])
        
        # Interpolate to get the desired number of points
        if len(landmarks) < num_points:
            landmarks = np.array(landmarks)
            t = np.linspace(0, 1, len(landmarks), endpoint=False)
            t_new = np.linspace(0, 1, num_points, endpoint=False)
            landmarks = np.array([np.interp(t_new, t, landmarks[:, i]) for i in range(2)]).T
        
        return landmarks.tolist()

    def align_garment(self, garment_img, person_img, category):
        garment_mask = self.get_mask(garment_img, category)
        person_mask = self.get_mask(person_img, category)
        
        garment_landmarks = self.generate_landmarks(garment_mask)
        person_landmarks = self.generate_landmarks(person_mask)
        
        src_points = np.array(garment_landmarks)
        dst_points = np.array(person_landmarks)
        
        # Ensure we have enough points for the transformation
        if len(src_points) < 4 or len(dst_points) < 4:
            print("Not enough points for transformation. Using simple resize.")
            return cv2.resize(np.array(garment_img), person_img.size[::-1])
        
        try:
            # Ensure src_points and dst_points have the same number of points
            min_points = min(len(src_points), len(dst_points))
            src_points = src_points[:min_points]
            dst_points = dst_points[:min_points]
            
            tform = PiecewiseAffineTransform()
            tform.estimate(src_points, dst_points)
            
            warped_garment = warp(np.array(garment_img), tform, output_shape=person_img.size[::-1])
            warped_garment = (warped_garment * 255).astype(np.uint8)
            
            # Apply the mask to the warped garment
            warped_mask = warp(garment_mask, tform, output_shape=person_img.size[::-1])
            warped_mask = (warped_mask > 0.5).astype(np.uint8)
            
            result = np.array(person_img)
            result[warped_mask > 0] = warped_garment[warped_mask > 0]
            
            return result
        except Exception as e:
            print(f"Transformation failed: {e}")
            print("Using simple resize as fallback.")
            return cv2.resize(np.array(garment_img), person_img.size[::-1])


    
if __name__ == "__main__":
    masker = Masking()
    garment_path = "images/b9.png"
    person_path = "TEST/mota.jpg"
    output_path = "images/output_image.jpg"
    category = "upper_body"  # Change as needed
    
    try:
        garment_img = Image.open(garment_path).convert("RGB")
        person_img = Image.open(person_path).convert("RGB")
        
        result = masker.align_garment(garment_img, person_img, category)
        
        # Convert result to RGB if it's RGBA
        if result.shape[2] == 4:
            result = result[:, :, :3]
        
        Image.fromarray(result).save(output_path)
        print(f"Aligned garment saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()