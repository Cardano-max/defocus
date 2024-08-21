import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from functools import wraps
from time import time
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from pathlib import Path
from skimage import measure, morphology
from scipy import ndimage
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} args:[{args}, {kw}] took: {te-ts:.4f} sec')
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
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

        # Load DeepLabV3 model for body segmentation
        self.deeplab_model = deeplabv3_resnet101(pretrained=True)
        self.deeplab_model.eval()
        if torch.cuda.is_available():
            self.deeplab_model.cuda()
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @timing
    def get_mask(self, img, category='upper_body'):
        img_resized = img.resize((384, 512), Image.LANCZOS)
        img_np = np.array(img_resized)
        
        parse_result, _ = self.parsing_model(img_resized)
        parse_array = np.array(parse_result)

        keypoints = self.openpose_model(img_resized)
        pose_data = np.array(keypoints["pose_keypoints_2d"]).reshape((-1, 2))

        # Check if the body is naked
        is_naked = self.is_body_naked(parse_array)

        if is_naked:
            mask = self.create_naked_body_mask(img)
        else:
            if category == 'upper_body':
                mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"]])
            elif category == 'lower_body':
                mask = np.isin(parse_array, [self.label_map["pants"], self.label_map["skirt"]])
            elif category == 'dresses':
                mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"], 
                                             self.label_map["pants"], self.label_map["skirt"]])
            else:
                raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

            mask = self.enhance_garment_mask(mask, parse_array, category)

        hand_mask = self.create_precise_hand_mask(img_np)
        arm_mask = np.isin(parse_array, [self.label_map["left_arm"], self.label_map["right_arm"]])
        
        # Combine hand and arm masks
        hand_arm_mask = np.logical_or(hand_mask, arm_mask)
        
        # Remove hand and arm regions from the mask if not naked
        if not is_naked:
            mask = np.logical_and(mask, np.logical_not(hand_arm_mask))

        # Apply final refinements
        final_mask = self.apply_final_refinements(mask)

        # Exclude face and include a small portion of the neck
        face_mask = np.isin(parse_array, [self.label_map["head"], self.label_map["hair"]])
        neck_mask = np.isin(parse_array, [self.label_map["neck"]])
        neck_mask_dilated = cv2.dilate(neck_mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1)
        final_mask = np.logical_and(final_mask, np.logical_not(face_mask))
        final_mask = np.logical_or(final_mask, np.logical_and(neck_mask_dilated, np.logical_not(neck_mask)))

        mask_pil = Image.fromarray((final_mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(img.size, Image.LANCZOS)
        
        return mask_pil

    def is_body_naked(self, parse_array):
        clothed_pixels = np.sum(np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"], 
                                                      self.label_map["pants"], self.label_map["skirt"]]))
        total_pixels = parse_array.size
        clothing_ratio = clothed_pixels / total_pixels
        return clothing_ratio < 0.1  # Adjust this threshold as needed

    def create_naked_body_mask(self, img):
        input_tensor = self.transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        
        with torch.no_grad():
            output = self.deeplab_model(input_tensor)['out'][0]
        
        output_predictions = output.argmax(0).byte().cpu().numpy()
        
        # Class 15 in DeepLabV3 corresponds to the human body
        body_mask = (output_predictions == 15).astype(np.uint8)
        
        # Remove small regions and fill holes
        body_mask = self.remove_small_regions(body_mask)
        body_mask = ndimage.binary_fill_holes(body_mask).astype(np.uint8)
        
        return body_mask

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
        
        # Dilate the hand mask slightly to ensure full coverage
        kernel = np.ones((5,5), np.uint8)
        hand_mask = cv2.dilate(hand_mask, kernel, iterations=1)
        
        return hand_mask > 0

    def enhance_garment_mask(self, initial_mask, parse_array, category):
        # Create a more inclusive garment region
        if category == 'upper_body':
            garment_labels = [self.label_map["upper_clothes"], self.label_map["dress"]]
        elif category == 'lower_body':
            garment_labels = [self.label_map["pants"], self.label_map["skirt"]]
        else:  # dresses
            garment_labels = [self.label_map["upper_clothes"], self.label_map["dress"], 
                              self.label_map["pants"], self.label_map["skirt"]]
        
        garment_region = np.isin(parse_array, garment_labels)
        
        # Combine initial mask with the parsing result
        enhanced_mask = np.logical_or(initial_mask, garment_region)
        
        # Fill holes in the mask
        enhanced_mask = ndimage.binary_fill_holes(enhanced_mask)
        
        # Remove small isolated regions
        enhanced_mask = self.remove_small_regions(enhanced_mask)
        
        # Dilate the mask slightly to extend beyond garment boundaries
        kernel = np.ones((5,5), np.uint8)
        enhanced_mask = cv2.dilate(enhanced_mask.astype(np.uint8), kernel, iterations=1)
        
        return enhanced_mask

    def apply_final_refinements(self, mask):
        # Smooth the edges
        mask = self.smooth_edges(mask, sigma=1.0)
        
        # Ensure the mask is binary
        mask = mask > 0.5
        
        # Fill any remaining small holes
        mask = ndimage.binary_fill_holes(mask)
        
        # Final dilation to ensure complete coverage
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        
        return mask

    def smooth_edges(self, mask, sigma=1.0):
        mask_float = mask.astype(float)
        mask_blurred = ndimage.gaussian_filter(mask_float, sigma=sigma)
        mask_smooth = (mask_blurred > 0.5).astype(np.uint8)
        return mask_smooth

    def remove_small_regions(self, mask, min_size=100):
        labeled, num_features = measure.label(mask, return_num=True)
        for i in range(1, num_features + 1):
            region = (labeled == i)
            if np.sum(region) < min_size:
                mask[region] = 0
        return mask

def process_images(input_folder, output_folder, category):
    masker = Masking()
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    image_files = list(Path(input_folder).glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')]
    
    for i, image_file in enumerate(image_files, 1):
        output_mask = Path(output_folder) / f"output_mask_{i}.png"
        output_masked = Path(output_folder) / f"output_masked_image_{i}.png"
        
        print(f"Processing image {i}/{len(image_files)}: {image_file.name}")
        
        input_img = Image.open(image_file).convert('RGB')
        
        mask = masker.get_mask(input_img, category=category)
        
        # Save the mask
        mask.save(str(output_mask))
        
        # Apply the mask to the original image
        masked_output = Image.composite(input_img, Image.new('RGB', input_img.size, (255, 255, 255)), mask)
        masked_output.save(str(output_masked))
        
        print(f"Mask saved to {output_mask}")
        print(f"Masked output saved to {output_masked}")
        print()


if __name__ == "__main__":
    input_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/in_im")
    output_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/output")
    category = "dresses"  # Change to "upper_body", "lower_body", or "dresses" as needed
    
    process_images(str(input_folder), str(output_folder), category)