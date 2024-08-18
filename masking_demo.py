import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from functools import wraps
from time import time
from Masking.preprocess.humanparsing.run_parsing import Parsing
from Masking.preprocess.openpose.run_openpose import OpenPose
from pathlib import Path
from skimage import measure, morphology
from scipy import ndimage
from PIL.Image import Resampling

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
        img_resized = img.resize((384, 512), Resampling.LANCZOS)
        img_np = np.array(img_resized)
        
        parse_result, _ = self.parsing_model(img_resized)
        parse_array = np.array(parse_result)

        keypoints = self.openpose_model(img_resized)
        pose_data = np.array(keypoints["pose_keypoints_2d"]).reshape((-1, 2))

        # Initial garment mask
        if category == 'upper_body':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"]])
        elif category == 'lower_body':
            mask = np.isin(parse_array, [self.label_map["pants"], self.label_map["skirt"]])
        elif category == 'dresses':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"], 
                                        self.label_map["pants"], self.label_map["skirt"]])
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

        # Improved body part detection
        body_mask = self.create_body_mask(img_np, parse_array, pose_data)
        hand_mask = self.create_hand_mask(img_np)
        
        # Combine body part masks
        combined_body_mask = np.logical_or(body_mask, hand_mask)
        
        # Remove body parts from the garment mask
        mask = np.logical_and(mask, np.logical_not(combined_body_mask))

        # Refine the mask
        mask = self.refine_mask(mask)
        mask = self.smooth_edges(mask)

        # Ensure the mask covers the full garment
        mask = self.fill_garment_gaps(mask, parse_array, category)

        # Additional refinement steps
        mask = self.post_process_mask(mask, img_np, parse_array)

        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(img.size, Resampling.LANCZOS)
        
        return np.array(mask_pil)

    def create_body_mask(self, image, parse_array, pose_data):
        body_parts = [self.label_map["left_arm"], self.label_map["right_arm"],
                    self.label_map["left_leg"], self.label_map["right_leg"],
                    self.label_map["head"], self.label_map["neck"]]
        body_mask = np.isin(parse_array, body_parts).astype(np.uint8) * 255
        
        # Use pose data to refine body mask
        h, w = image.shape[:2]
        for point in pose_data:
            if point[0] > 0 and point[1] > 0:
                x, y = int(point[0]), int(point[1])
                cv2.circle(body_mask, (x, y), 15, 255, -1)
        
        # Dilate the body mask to ensure complete coverage
        kernel = np.ones((7, 7), np.uint8)
        body_mask = cv2.dilate(body_mask, kernel, iterations=2)
        
        return body_mask > 0

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
                cv2.fillPoly(hand_mask, [hand_points], 1)
                
                # Dilate the hand mask to ensure complete coverage
                kernel = np.ones((7, 7), np.uint8)
                hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)
        
        return hand_mask > 0

    def refine_mask(self, mask):
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours and keep only the largest one
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_refined = np.zeros_like(mask_uint8)
            cv2.drawContours(mask_refined, [largest_contour], 0, 255, -1)
        else:
            mask_refined = mask_uint8
        
        return mask_refined > 0

    def smooth_edges(self, mask, sigma=2.0):
        mask_float = mask.astype(float)
        mask_blurred = ndimage.gaussian_filter(mask_float, sigma=sigma)
        mask_smooth = (mask_blurred > 0.5).astype(np.uint8)
        return mask_smooth

    def fill_garment_gaps(self, mask, parse_array, category):
        if category == 'upper_body':
            garment_labels = [self.label_map["upper_clothes"], self.label_map["dress"]]
        elif category == 'lower_body':
            garment_labels = [self.label_map["pants"], self.label_map["skirt"]]
        else:  # dresses
            garment_labels = [self.label_map["upper_clothes"], self.label_map["dress"], 
                              self.label_map["pants"], self.label_map["skirt"]]
        
        garment_region = np.isin(parse_array, garment_labels)
        
        # Use the garment region to fill gaps in the mask
        filled_mask = np.logical_or(mask, garment_region)
        
        # Remove small isolated regions
        filled_mask = self.remove_small_regions(filled_mask)
        
        return filled_mask

    def remove_small_regions(self, mask, min_size=200):
        labeled, num_features = measure.label(mask, return_num=True)
        for i in range(1, num_features + 1):
            region = (labeled == i)
            if np.sum(region) < min_size:
                mask[region] = 0
        return mask

    def post_process_mask(self, mask, image, parse_array):
        # Previous post-processing steps
        mask = ndimage.binary_fill_holes(mask)
        mask = morphology.remove_small_objects(mask, min_size=500)
        mask = self.smooth_edges(mask, sigma=1.5)
        
        # Consistency check with original parsing
        garment_parse = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"], 
                                            self.label_map["pants"], self.label_map["skirt"]])
        inconsistent = np.logical_xor(mask, garment_parse)
        
        # Prepare grabcut mask
        grabcut_mask = np.where(inconsistent, cv2.GC_PR_FGD, cv2.GC_FGD).astype(np.uint8)
        
        # Ensure image is in the correct format (8-bit, 3-channel)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        # Initialize background and foreground models
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        
        # Define ROI for grabcut
        rect = (0, 0, image.shape[1]-1, image.shape[0]-1)
        
        # Apply grabcut
        try:
            cv2.grabCut(image, grabcut_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        except cv2.error as e:
            print(f"GrabCut error: {str(e)}")
            print(f"Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"Mask shape: {grabcut_mask.shape}, dtype: {grabcut_mask.dtype}")
            return mask  # Return the original mask if grabcut fails
        
        # Update mask with grabcut results
        mask = np.where((grabcut_mask==2)|(grabcut_mask==0), 0, 1).astype('uint8')
        
        return mask

    @staticmethod
    def hole_fill(img):
        img = np.pad(img[1:-1, 1:-1], pad_width = 1, mode = 'constant', constant_values=0)
        img_copy = img.copy()
        mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

        cv2.floodFill(img, mask, (0, 0), 255)
        img_inverse = cv2.bitwise_not(img)
        dst = cv2.bitwise_or(img_copy, img_inverse)
        return dst

def process_images(input_folder, output_folder, category):
    masker = Masking()
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all image files from the input folder
    image_files = list(Path(input_folder).glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')]
    
    for i, image_file in enumerate(image_files, 1):
        output_mask = Path(output_folder) / f"output_sharp_mask_{i}.png"
        output_masked = Path(output_folder) / f"output_masked_image_white_bg_{i}.png"
        
        print(f"Processing image {i}/{len(image_files)}: {image_file.name}")
        
        input_img = Image.open(image_file)
        
        mask = masker.get_mask(input_img, category=category)
        
        Image.fromarray(mask).save(str(output_mask))
        
        # Create a white background image
        white_bg = Image.new('RGB', input_img.size, (255, 255, 255))
        
        # Convert input image to RGBA if it's not already
        if input_img.mode != 'RGBA':
            input_img = input_img.convert('RGBA')
        
        # Create a new image with white background and paste the masked input image
        masked_output = Image.composite(input_img, white_bg, Image.fromarray(mask))
        
        masked_output.save(str(output_masked))
        
        print(f"Mask saved to {output_mask}")
        print(f"Masked output with white background saved to {output_masked}")
        print()

if __name__ == "__main__":
    input_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/in_im")
    output_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/output")
    category = "dresses"  # Change this to "upper_body", "lower_body", or "dresses" as needed
    
    process_images(str(input_folder), str(output_folder), category)