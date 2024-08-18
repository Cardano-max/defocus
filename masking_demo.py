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
import torch
import torchvision.transforms as T
from skimage import draw, measure


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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        # Remove all body parts except clothes
        body_parts = [self.label_map["left_arm"], self.label_map["right_arm"],
                      self.label_map["left_leg"], self.label_map["right_leg"],
                      self.label_map["head"], self.label_map["neck"]]
        body_mask = np.isin(parse_array, body_parts)
        
        hand_mask = self.create_hand_mask(img_np)
        pose_mask = self.create_pose_mask(img_np)
        
        body_hand_pose_mask = np.logical_or(np.logical_or(body_mask, hand_mask), pose_mask)
        mask = np.logical_and(mask, np.logical_not(body_hand_pose_mask))

        mask = self.refine_mask(mask)
        mask = self.smooth_edges(mask)
        mask = self.remove_small_regions(mask)

        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize(img.size, Image.Resampling.LANCZOS)
        
        return np.array(mask_pil)

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
                cv2.fillPoly(hand_mask, [hand_points], 255)
                
                # Dilate the hand mask to ensure complete coverage
                kernel = np.ones((5,5), np.uint8)
                hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)
        
        return hand_mask > 0

    def create_pose_mask(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        pose_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            points = np.array([(int(l.x * image.shape[1]), int(l.y * image.shape[0])) for l in landmarks])
            
            # Create a more precise body outline
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(pose_mask, hull, 255)
            
            # Dilate the pose mask to ensure complete coverage
            kernel = np.ones((5,5), np.uint8)
            pose_mask = cv2.dilate(pose_mask, kernel, iterations=2)
        
        return pose_mask > 0

    def refine_mask(self, mask):
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Use more advanced morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # Use contour approximation for smoother edges
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.002 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            mask_refined = np.zeros_like(mask_uint8)
            cv2.drawContours(mask_refined, [approx], 0, 255, -1)
        else:
            mask_refined = mask_uint8
        
        return mask_refined > 0

    def smooth_edges(self, mask, sigma=1.5):
        mask_float = mask.astype(float)
        mask_blurred = ndimage.gaussian_filter(mask_float, sigma=sigma)
        mask_smooth = (mask_blurred > 0.5).astype(np.uint8)
        
        # Additional edge refinement using active contours
        contours = measure.find_contours(mask_smooth, 0.5)
        for contour in contours:
            rr, cc = draw.polygon(contour[:, 0], contour[:, 1], mask_smooth.shape)
            mask_smooth[rr, cc] = 1
        
        return mask_smooth

    def remove_small_regions(self, mask, min_size=100):
        labeled_mask, num_labels = ndimage.label(mask)
        sizes = ndimage.sum(mask, labeled_mask, range(1, num_labels + 1))
        mask_sizes = sizes < min_size
        remove_pixel = mask_sizes[labeled_mask - 1]
        mask[remove_pixel] = 0
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

def process_images(input_folder, output_folder, category, output_format='png'):
    masker = Masking()
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all image files from the input folder
    image_files = list(Path(input_folder).glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')]
    
    for i, image_file in enumerate(image_files, 1):
        output_mask = Path(output_folder) / f"output_sharp_mask_{i}.{output_format}"
        output_masked = Path(output_folder) / f"output_masked_image_white_bg_{i}.{output_format}"
        output_upscaled = Path(output_folder) / f"output_upscaled_8k_{i}.{output_format}"
        
        print(f"Processing image {i}/{len(image_files)}: {image_file.name}")
        
        input_img = Image.open(image_file).convert('RGB')
        
        mask = masker.get_mask(input_img, category=category)
        
        Image.fromarray(mask).save(str(output_mask))
        
        # Create a white background image
        white_bg = Image.new('RGB', input_img.size, (255, 255, 255))
        
        # Create a new image with white background and paste the masked input image
        masked_output = Image.composite(input_img, white_bg, Image.fromarray(mask))
        
        # Remove background using rembg for better results
        masked_output_removed_bg = remove(np.array(masked_output))
        masked_output_removed_bg = Image.fromarray(masked_output_removed_bg)
        
        if output_format.lower() == 'webp':
            masked_output_removed_bg.save(str(output_masked), format='WebP', lossless=True)
        else:
            masked_output_removed_bg.save(str(output_masked))
        
        # Upscale to 8K resolution using a simple bicubic interpolation
        target_size = (7680, 4320)  # 8K resolution
        upscaled_output = masked_output_removed_bg.resize(target_size, Image.BICUBIC)
        
        if output_format.lower() == 'webp':
            upscaled_output.save(str(output_upscaled), format='WebP', lossless=True)
        else:
            upscaled_output.save(str(output_upscaled))
        
        print(f"Mask saved to {output_mask}")
        print(f"Masked output with white background saved to {output_masked}")
        print(f"Upscaled 8K output saved to {output_upscaled}")
        print()

if __name__ == "__main__":
    input_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/Input_Images")
    output_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/output")
    category = "dresses"  # Change this to "upper_body", "lower_body", or "dresses" as needed
    output_format = "webp"  # Change this to "png" if you prefer PNG output
    
    process_images(str(input_folder), str(output_folder), category, output_format)