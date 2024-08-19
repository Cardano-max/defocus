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

import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from functools import wraps
from time import time
from Masking.preprocess.humanparsing.run_parsing import Parsing
from Masking.preprocess.openpose.run_openpose import OpenPose
from skimage import measure, morphology
from scipy import ndimage

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
        img_resized = img.resize((384, 512), Image.LANCZOS)
        img_np = np.array(img_resized)
        
        parse_result, _ = self.parsing_model(img_resized)
        parse_array = np.array(parse_result)

        keypoints = self.openpose_model(img_resized)
        pose_data = np.array(keypoints["pose_keypoints_2d"]).reshape((-1, 2))

        if category == 'upper_body':
            garment_labels = [self.label_map["upper_clothes"], self.label_map["dress"]]
        elif category == 'lower_body':
            garment_labels = [self.label_map["pants"], self.label_map["skirt"]]
        elif category == 'dresses':
            garment_labels = [self.label_map["upper_clothes"], self.label_map["dress"], 
                              self.label_map["pants"], self.label_map["skirt"]]
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

        mask = np.isin(parse_array, garment_labels)

        body_mask = self.create_body_mask(img_np, parse_array, pose_data)
        hand_mask = self.create_hand_mask(img_np)
        
        combined_body_mask = np.logical_or(body_mask, hand_mask)
        mask = np.logical_and(mask, np.logical_not(combined_body_mask))

        mask = self.refine_mask(mask, img_np, parse_array, category)
        mask = self.smooth_edges(mask)
        mask = self.post_process_mask(mask, img_np, parse_array)

        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(img.size, Image.LANCZOS)
        
        return np.array(mask_pil)

    def create_body_mask(self, image, parse_array, pose_data):
        body_parts = [self.label_map["left_arm"], self.label_map["right_arm"],
                      self.label_map["left_leg"], self.label_map["right_leg"],
                      self.label_map["head"], self.label_map["neck"]]
        body_mask = np.isin(parse_array, body_parts)
        
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            h, w = image.shape[:2]
            for landmark in results.pose_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(body_mask.astype(np.uint8), (x, y), 10, 1, -1)
        
        return body_mask

    def create_hand_mask(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        hand_mask = np.zeros(image.shape[:2], dtype=bool)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    hand_points.append([x, y])
                hand_points = np.array(hand_points, dtype=np.int32)
                cv2.fillPoly(hand_mask, [hand_points], True)
        
        return hand_mask

    def refine_mask(self, mask, image, parse_array, category):
        # Use GrabCut for initial refinement
        grabcut_mask = np.where(mask, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (0,0,image.shape[1],image.shape[0])
        cv2.grabCut(image, grabcut_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        
        mask = np.where((grabcut_mask==2)|(grabcut_mask==0), 0, 1).astype(bool)
        
        # Fill holes and remove small objects
        mask = ndimage.binary_fill_holes(mask)
        mask = morphology.remove_small_objects(mask, min_size=500)
        
        return mask

    def smooth_edges(self, mask, sigma=1.5):
        mask_float = mask.astype(float)
        mask_blurred = ndimage.gaussian_filter(mask_float, sigma=sigma)
        mask_smooth = (mask_blurred > 0.5).astype(bool)
        return mask_smooth

    def post_process_mask(self, mask, image, parse_array):
        # Use the parsing results to refine the edges
        garment_parse = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"], 
                                              self.label_map["pants"], self.label_map["skirt"]])
        mask = np.logical_or(mask, garment_parse)
        
        # Final smoothing
        mask = self.smooth_edges(mask, sigma=0.5)
        
        return mask

def process_images(input_folder, output_folder, category):
    masker = Masking()
    
    image_files = list(Path(input_folder).glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')]
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    for i, image_file in enumerate(image_files, 1):
        print(f"Processing image {i}/{len(image_files)}: {image_file.name}")
        
        input_img = Image.open(image_file)
        mask = masker.get_mask(input_img, category=category)
        
        mask_pil = Image.fromarray(mask)
        mask_pil.save(Path(output_folder) / f"output_mask_{i}.png")
        
        masked_output = Image.composite(input_img.convert('RGBA'), 
                                        Image.new('RGBA', input_img.size, (255,255,255,255)), 
                                        Image.fromarray(mask))
        masked_output.save(Path(output_folder) / f"output_masked_{i}.png")
        
        print(f"Mask and masked output saved for {image_file.name}")
        print()

if __name__ == "__main__":
    input_folder = Path("path/to/input/folder")
    output_folder = Path("path/to/output/folder")
    category = "dresses"  # Change as needed
    
    process_images(input_folder, output_folder, category)