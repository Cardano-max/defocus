import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from functools import wraps
from time import time
from pathlib import Path
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
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    @timing
    def get_mask(self, img, category='upper_body'):
        img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
        img_np = np.array(img_resized)
        
        # Use MediaPipe's Selfie Segmentation for initial person segmentation
        results_segmentation = self.selfie_segmentation.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        person_mask = results_segmentation.segmentation_mask > 0.1
        
        # Refine the mask using pose estimation
        refined_mask = self.refine_mask_with_pose(img_np, person_mask, category)
        
        # Post-process the mask
        refined_mask = self.post_process_mask(refined_mask)

        # Resize mask back to original image size
        mask_pil = Image.fromarray(refined_mask.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize(img.size, Image.Resampling.LANCZOS)
        
        return np.array(mask_pil)

    def refine_mask_with_pose(self, image, initial_mask, category):
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        refined_mask = initial_mask.copy()
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            height, width = initial_mask.shape[:2]
            
            # Define key points for different body parts
            shoulder_l = np.array([landmarks[11].x, landmarks[11].y])
            shoulder_r = np.array([landmarks[12].x, landmarks[12].y])
            hip_l = np.array([landmarks[23].x, landmarks[23].y])
            hip_r = np.array([landmarks[24].x, landmarks[24].y])
            
            # Create a more precise body outline based on category
            body_mask = np.zeros((height, width), dtype=np.uint8)
            
            if category in ['upper_body', 'dresses']:
                cv2.fillConvexPoly(body_mask, np.array([
                    (shoulder_l * [width, height]).astype(int),
                    (shoulder_r * [width, height]).astype(int),
                    (hip_r * [width, height]).astype(int),
                    (hip_l * [width, height]).astype(int)
                ]), 1)
            
            if category in ['lower_body', 'dresses']:
                ankle_l = np.array([landmarks[27].x, landmarks[27].y])
                ankle_r = np.array([landmarks[28].x, landmarks[28].y])
                cv2.fillConvexPoly(body_mask, np.array([
                    (hip_l * [width, height]).astype(int),
                    (hip_r * [width, height]).astype(int),
                    (ankle_r * [width, height]).astype(int),
                    (ankle_l * [width, height]).astype(int)
                ]), 1)
            
            # Refine the initial mask
            refined_mask = np.logical_and(refined_mask, body_mask)
        
        return refined_mask

    def post_process_mask(self, mask, min_size=100, kernel_size=5):
        # Remove small objects
        mask_cleaned = ndimage.binary_opening(mask, structure=np.ones((3,3)))
        
        # Fill holes
        mask_filled = ndimage.binary_fill_holes(mask_cleaned)
        
        # Smooth edges
        mask_smooth = ndimage.gaussian_filter(mask_filled.astype(float), sigma=1)
        mask_smooth = (mask_smooth > 0.5).astype(np.uint8)
        
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
        
        if output_format.lower() == 'webp':
            masked_output.save(str(output_masked), format='WebP', lossless=True)
        else:
            masked_output.save(str(output_masked))
        
        print(f"Mask saved to {output_mask}")
        print(f"Masked output saved to {output_masked}")
        print()

if __name__ == "__main__":
    input_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/in_im")
    output_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/output")
    category = "dresses"  # Change this to "upper_body", "lower_body", or "dresses" as needed
    output_format = "png"
    
    process_images(str(input_folder), str(output_folder), category, output_format)