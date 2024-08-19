import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from functools import wraps
from time import time
from Masking.preprocess.humanparsing.run_parsing import Parsing
from Masking.preprocess.openpose.run_openpose import OpenPose
from pathlib import Path
from skimage import measure, morphology, segmentation, feature
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

        # Initialize HandLandmarker
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.99,
            min_hand_presence_confidence=0.99,
            min_tracking_confidence=0.99
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

    @timing
    def get_mask(self, img, category='upper_body'):
        img_resized = img.resize((384, 512), Image.Resampling.LANCZOS)
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

        # Detect hands
        hand_mask = self.create_hand_mask(img_np)
        
        # Enhance garment-body differentiation
        mask = self.enhance_garment_body_diff(mask, parse_array, pose_data, category)
        
        # Refine the mask
        mask = self.refine_mask(mask)
        
        # Apply advanced smoothing
        mask = self.advanced_smooth(mask)

        # Ensure the mask covers the full garment
        mask = self.fill_garment_gaps(mask, parse_array, category)

        # Additional refinement steps
        mask = self.post_process_mask(mask)

        # Ensure hands are always unmasked
        mask[hand_mask] = 0

        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(img.size, Image.Resampling.LANCZOS)
        
        return np.array(mask_pil)

    def create_hand_mask(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        detection_result = self.hand_landmarker.detect(mp_image)
        
        hand_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                hand_points = []
                for landmark in hand_landmarks:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    hand_points.append([x, y])
                hand_points = np.array(hand_points, dtype=np.int32)
                cv2.fillConvexPoly(hand_mask, hand_points, 1)
                
                # Dilate the hand mask to ensure complete coverage
                kernel = np.ones((7, 7), np.uint8)
                hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)
        
        return hand_mask > 0

    def enhance_garment_body_diff(self, mask, parse_array, pose_data, category):
        # Use pose estimation to refine garment boundaries
        body_parts = [self.label_map["left_arm"], self.label_map["right_arm"],
                      self.label_map["left_leg"], self.label_map["right_leg"],
                      self.label_map["head"], self.label_map["neck"]]
        body_mask = np.isin(parse_array, body_parts)
        
        # Create a distance map from body keypoints
        distance_map = np.zeros_like(mask, dtype=float)
        for point in pose_data:
            if point[0] > 0 and point[1] > 0:
                y, x = int(point[1]), int(point[0])
                distance_map += ndimage.distance_transform_edt(np.ones_like(mask)) - ndimage.distance_transform_edt(np.zeros_like(mask))
        
        # Normalize distance map
        distance_map = (distance_map - distance_map.min()) / (distance_map.max() - distance_map.min())
        
        # Combine mask, body mask, and distance map
        refined_mask = np.logical_and(mask, np.logical_not(body_mask))
        refined_mask = np.logical_or(refined_mask, distance_map < 0.1)  # Adjust threshold as needed
        
        return refined_mask

    def refine_mask(self, mask):
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Apply more aggressive morphological operations
        kernel = np.ones((7,7), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return mask_uint8 > 0

    def advanced_smooth(self, mask, sigma=2.0):
        # Convert mask to uint8 for bilateralFilter
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Apply bilateral filter for edge-preserving smoothing
        mask_smoothed = cv2.bilateralFilter(mask_uint8, 9, 75, 75)
        
        # Convert back to float for Gaussian smoothing
        mask_float = mask_smoothed.astype(float) / 255.0
        
        # Apply additional Gaussian smoothing
        mask_smoothed = ndimage.gaussian_filter(mask_float, sigma=sigma)
        
        # Threshold to get binary mask
        mask_smooth = (mask_smoothed > 0.5).astype(np.uint8)
        
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
        
        # Remove small isolated regions and fill small holes
        filled_mask = self.remove_small_regions(filled_mask)
        filled_mask = ndimage.binary_fill_holes(filled_mask)
        
        return filled_mask

    def remove_small_regions(self, mask, min_size=300):
        labeled, num_features = measure.label(mask, return_num=True)
        for i in range(1, num_features + 1):
            region = (labeled == i)
            if np.sum(region) < min_size:
                mask[region] = 0
        return mask

    def post_process_mask(self, mask):
        # Apply watershed algorithm for more precise segmentation
        distance = ndimage.distance_transform_edt(mask)
        local_max = feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=mask)
        markers = measure.label(local_max)
        labels = segmentation.watershed(-distance, markers, mask=mask)
        
        # Smooth boundaries
        mask = self.advanced_smooth(labels > 0, sigma=1.5)
        
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