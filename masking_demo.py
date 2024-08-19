import numpy as np
import cv2
from PIL import Image
from functools import wraps
from time import time
from Masking.preprocess.humanparsing.run_parsing import Parsing
from Masking.preprocess.openpose.run_openpose import OpenPose
from pathlib import Path
from skimage import measure, morphology
from scipy import ndimage

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
        self.label_map = {
            "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
            "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
            "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
            "bag": 16, "scarf": 17, "neck": 18
        }

    @timing
    def get_mask(self, img, category='upper_body'):
        img_resized = img.resize((384, 512), Image.LANCZOS)
        img_np = np.array(img_resized)
        
        parse_result, _ = self.parsing_model(img_resized)
        parse_array = np.array(parse_result)

        keypoints = self.openpose_model(img_resized)
        pose_data = np.array(keypoints["pose_keypoints_2d"]).reshape((-1, 2))

        if category == 'upper_body':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"]])
        elif category == 'lower_body':
            mask = np.isin(parse_array, [self.label_map["pants"], self.label_map["skirt"]])
        elif category == 'dresses':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"], 
                                         self.label_map["pants"], self.label_map["skirt"]])
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

        # Create hand mask using OpenPose data
        hand_mask = self.create_hand_mask(pose_data, img_np.shape[:2])
        
        # Create arm mask (including exposed skin)
        arm_mask = np.isin(parse_array, [self.label_map["left_arm"], self.label_map["right_arm"]])
        
        # Create neck mask
        neck_mask = parse_array == self.label_map["neck"]
        
        # Combine garment, arm, and neck masks
        full_garment_mask = np.logical_or(mask, arm_mask)
        full_garment_mask = np.logical_or(full_garment_mask, neck_mask)
        
        # Refine the full garment mask
        full_garment_mask = self.refine_mask(full_garment_mask)
        full_garment_mask = self.smooth_edges(full_garment_mask, sigma=1.0)
        full_garment_mask = self.fill_garment_gaps(full_garment_mask, parse_array, category)
        full_garment_mask = self.expand_mask(full_garment_mask)

        # Unmask hand regions from the full garment mask
        final_mask = np.logical_and(full_garment_mask, np.logical_not(hand_mask))

        final_mask_pil = Image.fromarray((final_mask * 255).astype(np.uint8))
        final_mask_pil = final_mask_pil.resize(img.size, Image.LANCZOS)
        
        return np.array(final_mask_pil)

    def create_hand_mask(self, pose_data, shape):
        hand_mask = np.zeros(shape, dtype=np.uint8)
        
        # OpenPose hand keypoints indices
        left_hand_indices = [4, 3, 2, 1]  # Left wrist to left elbow
        right_hand_indices = [7, 6, 5, 1]  # Right wrist to right elbow
        
        def draw_hand(indices):
            points = pose_data[indices]
            valid_points = points[points[:, 0] != 0]
            if len(valid_points) >= 2:
                wrist = valid_points[0]
                elbow = valid_points[-1]
                hand_direction = wrist - elbow
                hand_length = np.linalg.norm(hand_direction) * 0.3  # Adjust this factor to change hand size
                hand_end = wrist + (hand_direction / np.linalg.norm(hand_direction)) * hand_length
                cv2.line(hand_mask, tuple(wrist.astype(int)), tuple(hand_end.astype(int)), 1, 15)
                cv2.circle(hand_mask, tuple(wrist.astype(int)), 20, 1, -1)
        
        draw_hand(left_hand_indices)
        draw_hand(right_hand_indices)
        
        # Dilate the hand mask to ensure full coverage
        kernel = np.ones((7, 7), np.uint8)
        hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)
        
        return hand_mask > 0

    def refine_mask(self, mask):
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours and keep only the largest one
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_refined = np.zeros_like(mask_uint8)
            cv2.drawContours(mask_refined, [largest_contour], 0, 255, 3)  # Thicker outline
            mask_refined = cv2.fillPoly(mask_refined, [largest_contour], 255)
        else:
            mask_refined = mask_uint8
        
        return mask_refined > 0

    def smooth_edges(self, mask, sigma=1.0):
        mask_float = mask.astype(float)
        mask_blurred = ndimage.gaussian_filter(mask_float, sigma=sigma)
        mask_smooth = (mask_blurred > 0.5).astype(np.uint8)
        return mask_smooth

    def fill_garment_gaps(self, mask, parse_array, category):
        if category == 'upper_body':
            garment_labels = [self.label_map["upper_clothes"], self.label_map["dress"], 
                              self.label_map["left_arm"], self.label_map["right_arm"], self.label_map["neck"]]
        elif category == 'lower_body':
            garment_labels = [self.label_map["pants"], self.label_map["skirt"]]
        else:  # dresses
            garment_labels = [self.label_map["upper_clothes"], self.label_map["dress"], 
                              self.label_map["pants"], self.label_map["skirt"],
                              self.label_map["left_arm"], self.label_map["right_arm"], self.label_map["neck"]]
        
        garment_region = np.isin(parse_array, garment_labels)
        
        # Use the garment region to fill gaps in the mask
        filled_mask = np.logical_or(mask, garment_region)
        
        # Remove small isolated regions
        filled_mask = self.remove_small_regions(filled_mask)
        
        return filled_mask

    def remove_small_regions(self, mask, min_size=100):
        labeled, num_features = measure.label(mask, return_num=True)
        for i in range(1, num_features + 1):
            region = (labeled == i)
            if np.sum(region) < min_size:
                mask[region] = 0
        return mask

    def expand_mask(self, mask, expansion=3):
        kernel = np.ones((expansion, expansion), np.uint8)
        expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return expanded_mask > 0

def process_images(input_folder, output_folder, category):
    masker = Masking()
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    image_files = list(Path(input_folder).glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')]
    
    for i, image_file in enumerate(image_files, 1):
        output_mask = Path(output_folder) / f"output_sharp_mask_{i}.png"
        output_masked = Path(output_folder) / f"output_masked_image_white_bg_{i}.png"
        
        print(f"Processing image {i}/{len(image_files)}: {image_file.name}")
        
        input_img = Image.open(image_file)
        
        mask = masker.get_mask(input_img, category=category)
        
        Image.fromarray(mask).save(str(output_mask))
        
        white_bg = Image.new('RGB', input_img.size, (255, 255, 255))
        
        if input_img.mode != 'RGBA':
            input_img = input_img.convert('RGBA')
        
        masked_output = Image.composite(input_img, white_bg, Image.fromarray(mask))
        
        masked_output.save(str(output_masked))
        
        print(f"Mask saved to {output_mask}")
        print(f"Masked output with white background saved to {output_masked}")
        print()

if __name__ == "__main__":
    input_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/in_im")
    output_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/output")
    category = "dresses"  # Change to "upper_body", "lower_body", or "dresses" as needed
    
    process_images(str(input_folder), str(output_folder), category)