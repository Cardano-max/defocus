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
from rembg import remove

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
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
        self.label_map = {
            "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
            "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
            "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
            "bag": 16, "scarf": 17, "neck": 18
        }

    @timing
    def get_mask(self, img, category='upper_body'):
        img_resized = img.resize((384, 512), Image.Resampling.LANCZOS)
        img_np = np.array(img_resized)
        
        parse_result, _ = self.parsing_model(img_resized)
        parse_array = np.array(parse_result)

        keypoints = self.openpose_model(img_resized)
        pose_data = np.array(keypoints["pose_keypoints_2d"]).reshape((-1, 2))

        if category == 'upper_body':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"]])
        elif category == 'lower_body':
            mask = np.isin(parse_array, [self.label_map["pants"], self.label_map["skirt"]])
        elif category == 'dresses':
            mask = np.isin(parse_array, [self.label_map["dress"], self.label_map["upper_clothes"], self.label_map["pants"], self.label_map["skirt"]])
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

        # Refine the mask using pose estimation
        refined_mask = self.refine_mask_with_pose(img_np, mask)
        
        # Remove small isolated regions and fill holes
        refined_mask = self.post_process_mask(refined_mask)

        # Resize mask back to original image size
        mask_pil = Image.fromarray(refined_mask.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize(img.size, Image.Resampling.LANCZOS)
        
        return np.array(mask_pil)

    def refine_mask_with_pose(self, image, initial_mask):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        refined_mask = initial_mask.copy()
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Define key points for different body parts
            shoulder_l = np.array([landmarks[11].x, landmarks[11].y])
            shoulder_r = np.array([landmarks[12].x, landmarks[12].y])
            hip_l = np.array([landmarks[23].x, landmarks[23].y])
            hip_r = np.array([landmarks[24].x, landmarks[24].y])
            
            # Create a more precise body outline
            height, width = initial_mask.shape[:2]
            body_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Upper body
            cv2.fillConvexPoly(body_mask, np.array([
                (shoulder_l * [width, height]).astype(int),
                (shoulder_r * [width, height]).astype(int),
                (hip_r * [width, height]).astype(int),
                (hip_l * [width, height]).astype(int)
            ]), 1)
            
            # Refine the initial mask
            refined_mask = np.logical_and(refined_mask, body_mask)
        
        return refined_mask

    def post_process_mask(self, mask, min_size=100, kernel_size=5):
        # Remove small objects
        mask_cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)
        
        # Fill holes
        mask_filled = ndimage.binary_fill_holes(mask_cleaned)
        
        # Smooth edges
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_smooth = cv2.morphologyEx(mask_filled.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        return mask_smooth

def process_images(input_folder, output_folder, category, output_format='png'):
    masker = Masking()
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    image_files = list(Path(input_folder).glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')]
    
    for i, image_file in enumerate(image_files, 1):
        output_mask = Path(output_folder) / f"output_mask_{i}.{output_format}"
        output_masked = Path(output_folder) / f"output_masked_{i}.{output_format}"
        
        print(f"Processing image {i}/{len(image_files)}: {image_file.name}")
        
        input_img = Image.open(image_file).convert('RGB')
        
        mask = masker.get_mask(input_img, category=category)
        
        Image.fromarray(mask).save(str(output_mask))
        
        # Create a new image with transparent background where the mask is 0
        masked_output = input_img.copy()
        masked_output.putalpha(Image.fromarray(mask))
        
        # Remove background using rembg for better results
        masked_output_removed_bg = remove(np.array(masked_output))
        masked_output_removed_bg = Image.fromarray(masked_output_removed_bg)
        
        if output_format.lower() == 'webp':
            masked_output_removed_bg.save(str(output_masked), format='WebP', lossless=True)
        else:
            masked_output_removed_bg.save(str(output_masked))
        
        print(f"Mask saved to {output_mask}")
        print(f"Masked output saved to {output_masked}")
        print()

if __name__ == "__main__":
    input_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/in_im")
    output_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/output")
    category = "dresses"  # Change this to "upper_body", "lower_body", or "dresses" as needed
    output_format = "png"
    
    process_images(str(input_folder), str(output_folder), category, output_format)