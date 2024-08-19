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
from skimage import measure, morphology, filters, segmentation, color
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

        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.8,
            min_hand_presence_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

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

        hand_mask = self.create_precise_hand_mask(img_np)
        arm_mask = np.isin(parse_array, [self.label_map["left_arm"], self.label_map["right_arm"]])
        
        # Combine hand and arm masks
        hand_arm_mask = np.logical_or(hand_mask, arm_mask)
        
        # Refine the garment mask
        mask = self.refine_garment_mask(mask, img_np)
        
        # Handle shadows and edge cases
        mask = self.handle_shadows_and_edges(mask, img_np)
        
        # Remove hand and arm regions from the garment mask
        mask = np.logical_and(mask, np.logical_not(hand_arm_mask))

        # Final refinement
        mask = self.final_refinement(mask)

        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(img.size, Image.LANCZOS)
        
        return np.array(mask_pil)

    def create_precise_hand_mask(self, image):
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
                cv2.fillPoly(hand_mask, [hand_points], 1)
        
        return hand_mask > 0

    def refine_garment_mask(self, mask, image):
        # Convert mask to uint8
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Use grab cut for better segmentation
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        mask_grabcut = np.where((mask_uint8==0)|(mask_uint8==1), 0, 1).astype('uint8')
        cv2.grabCut(image, mask_grabcut, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        
        # Refine the mask
        mask_refined = np.where((mask_grabcut==2)|(mask_grabcut==0), 0, 1).astype('uint8')
        
        return mask_refined

    def handle_shadows_and_edges(self, mask, image):
        # Convert image to LAB color space
        lab = color.rgb2lab(image)
        l_channel = lab[:,:,0]
        
        # Apply adaptive thresholding to L channel
        thresh = filters.threshold_local(l_channel, block_size=51, offset=10)
        binary = l_channel > thresh
        
        # Combine with original mask
        combined_mask = np.logical_or(mask, binary)
        
        # Apply edge detection
        edges = filters.sobel(l_channel)
        
        # Use watershed algorithm to refine boundaries
        distance = ndimage.distance_transform_edt(combined_mask)
        local_maxi = filters.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=combined_mask)
        markers = measure.label(local_maxi)
        labels = segmentation.watershed(-distance, markers, mask=combined_mask)
        
        # Combine the results
        refined_mask = (labels > 0).astype(np.uint8)
        
        # Use the edges to further refine the mask
        refined_mask = np.logical_or(refined_mask, edges > filters.threshold_otsu(edges))
        
        return refined_mask

    def final_refinement(self, mask):
        # Remove small holes
        mask = morphology.remove_small_holes(mask, area_threshold=100)
        
        # Remove small objects
        mask = morphology.remove_small_objects(mask, min_size=100)
        
        # Apply morphological closing to smooth edges
        kernel = morphology.disk(2)
        mask = morphology.binary_closing(mask, kernel)
        
        # Smooth edges
        mask = filters.gaussian(mask, sigma=0.5)
        mask = (mask > 0.5).astype(np.uint8)
        
        return mask

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