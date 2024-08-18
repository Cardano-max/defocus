import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from functools import wraps
from time import time
from Masking.preprocess.humanparsing.run_parsing import Parsing
from Masking.preprocess.openpose.run_openpose import OpenPose
from pathlib import Path
from skimage import measure, morphology, segmentation, feature
from scipy import ndimage
from PIL.Image import Resampling
from scipy import ndimage as ndi


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
        try:
            print(f"Processing image for category: {category}")
            img_resized = img.resize((384, 512), Resampling.LANCZOS)
            img_np = np.array(img_resized)
            
            print("Running parsing model...")
            parse_result, _ = self.parsing_model(img_resized)
            parse_array = np.array(parse_result)

            print("Running OpenPose model...")
            keypoints = self.openpose_model(img_resized)
            pose_data = np.array(keypoints["pose_keypoints_2d"]).reshape((-1, 2))

            print("Creating initial mask...")
            if category == 'upper_body':
                mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"]])
            elif category == 'lower_body':
                mask = np.isin(parse_array, [self.label_map["pants"], self.label_map["skirt"]])
            elif category == 'dresses':
                mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"], 
                                            self.label_map["pants"], self.label_map["skirt"]])
            else:
                raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

            print("Creating body mask...")
            body_mask = self.create_body_mask(img_np, parse_array)
            print("Creating hand mask...")
            hand_mask = self.create_hand_mask(img_np)
            
            combined_body_mask = np.logical_or(body_mask, hand_mask)
            mask = np.logical_and(mask, np.logical_not(combined_body_mask))

            print("Refining mask...")
            mask = self.refine_mask(mask)
            print("Smoothing edges...")
            mask = self.smooth_edges(mask)
            print("Filling garment gaps...")
            mask = self.fill_garment_gaps(mask, parse_array, category)
            print("Post-processing mask...")
            mask = self.post_process_mask(mask)
            mask = np.logical_and(mask, np.logical_not(hand_mask))

            print("Applying GrabCut...")
            mask = self.apply_grabcut(img_np, mask)
            print("Final refinement...")
            mask = self.final_refinement(mask)

            print("Creating final mask image...")
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(img.size, Resampling.LANCZOS)
            
            print("Mask creation completed.")
            return np.array(mask_pil)
        except Exception as e:
            print(f"Error in get_mask: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def create_body_mask(self, image, parse_array):
        body_parts = [self.label_map["left_arm"], self.label_map["right_arm"],
                    self.label_map["left_leg"], self.label_map["right_leg"],
                    self.label_map["head"], self.label_map["neck"]]
        body_mask = np.isin(parse_array, body_parts).astype(np.uint8) * 255
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            h, w = image.shape[:2]
            for landmark in results.pose_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(body_mask, (x, y), 20, 255, -1)
        
        kernel = np.ones((9, 9), np.uint8)
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
                
                hull = cv2.convexHull(hand_points)
                cv2.fillConvexPoly(hand_mask, hull, 255)
                
                kernel = np.ones((15, 15), np.uint8)
                hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)
        
        return hand_mask > 0

    def refine_mask(self, mask):
        mask_uint8 = mask.astype(np.uint8) * 255
        
        kernel = np.ones((7, 7), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=3)
        
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_refined = np.zeros_like(mask_uint8)
            cv2.drawContours(mask_refined, [largest_contour], 0, 255, -1)
        else:
            mask_refined = mask_uint8
        
        return mask_refined > 0

    def smooth_edges(self, mask, sigma=2.5):
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
        filled_mask = np.logical_or(mask, garment_region)
        filled_mask = self.remove_small_regions(filled_mask)
        
        return filled_mask

    def remove_small_regions(self, mask, min_size=300):
        labeled, num_features = measure.label(mask, return_num=True)
        for i in range(1, num_features + 1):
            region = (labeled == i)
            if np.sum(region) < min_size:
                mask[region] = 0
        return mask

    def post_process_mask(self, mask):
        mask = ndimage.binary_fill_holes(mask)
        mask = morphology.remove_small_objects(mask, min_size=500)
        mask = self.smooth_edges(mask, sigma=1.5)
        return mask

    def apply_grabcut(self, image, mask):
        mask_gc = np.zeros(image.shape[:2], np.uint8)
        mask_gc[mask > 0] = cv2.GC_PR_FGD
        mask_gc[mask == 0] = cv2.GC_PR_BGD
        
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        mask_gc, _, _ = cv2.grabCut(image, mask_gc, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        
        return np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype('uint8')

    def final_refinement(self, mask):
        try:
            print("Starting final refinement...")
            distance = ndi.distance_transform_edt(mask)
            print("Distance transform completed.")
            
            # Remove the 'indices' parameter
            local_maxi = feature.peak_local_max(distance, footprint=np.ones((3, 3)), labels=mask)
            print("Local maxima found.")
            
            markers = np.zeros(distance.shape, dtype=bool)
            markers[tuple(local_maxi.T)] = True
            markers = measure.label(markers)
            print("Markers created.")
            
            labels = segmentation.watershed(-distance, markers, mask=mask)
            print("Watershed segmentation completed.")
            
            refined_mask = np.zeros_like(mask)
            for label in np.unique(labels):
                if label == 0:
                    continue
                region = labels == label
                if np.sum(region) > 1000:  # Adjust this threshold as needed
                    refined_mask = np.logical_or(refined_mask, region)
            
            print("Final refinement completed.")
            return refined_mask
        except Exception as e:
            print(f"Error in final_refinement: {str(e)}")
            import traceback
            traceback.print_exc()
            return mask  # Return the original mask if refinement fails

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
    category = "dresses"  # Change this to "upper_body", "lower_body", or "dresses" as needed
    
    process_images(str(input_folder), str(output_folder), category)