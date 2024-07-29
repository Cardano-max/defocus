import numpy as np
import torch
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from Masking.preprocess.openpose.run_openpose import OpenPose

class Masking:
    def __init__(self):
        self.processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")
        self.openpose = OpenPose(gpu_id=0)
        self.label_map = {
            "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
            "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
            "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
            "bag": 16, "scarf": 17
        }

    def get_body_mask(self, parse_array, category):
        if category == 'upper_body':
            return ((parse_array == self.label_map["upper_clothes"]) | 
                    (parse_array == self.label_map["dress"])).astype(np.uint8) * 255
        elif category == 'lower_body':
            return ((parse_array == self.label_map["pants"]) | 
                    (parse_array == self.label_map["skirt"]) | 
                    (parse_array == self.label_map["left_leg"]) | 
                    (parse_array == self.label_map["right_leg"])).astype(np.uint8) * 255
        elif category == 'dresses':
            return ((parse_array == self.label_map["dress"]) | 
                    (parse_array == self.label_map["upper_clothes"]) | 
                    (parse_array == self.label_map["pants"]) | 
                    (parse_array == self.label_map["skirt"])).astype(np.uint8) * 255
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

    def get_arm_mask(self, pose_data, shape):
        arm_mask = np.zeros(shape, dtype=np.uint8)
        right_arm = pose_data[[2, 3, 4]]  # Right shoulder, elbow, wrist
        left_arm = pose_data[[5, 6, 7]]   # Left shoulder, elbow, wrist
        
        for arm in [right_arm, left_arm]:
            pts = arm.astype(np.int32)
            cv2.fillPoly(arm_mask, [pts], 255)
            for i in range(3):
                cv2.circle(arm_mask, tuple(pts[i]), 30, 255, -1)
        
        return cv2.dilate(arm_mask, np.ones((20, 20), np.uint8), iterations=2)

    def get_mask(self, image, category='upper_body', width=384, height=512):
        # Ensure image is in RGB mode and resize
        image = image.convert('RGB').resize((width, height), Image.NEAREST)
        np_image = np.array(image)

        # Get segmentation
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        parse_array = upsampled_logits.argmax(dim=1)[0].numpy()

        # Get pose estimation
        keypoints = self.openpose(np_image)
        pose_data = np.array(keypoints["pose_keypoints_2d"]).reshape((-1, 2))

        # Create body mask based on category
        body_mask = self.get_body_mask(parse_array, category)

        # Create arm mask
        arm_mask = self.get_arm_mask(pose_data, body_mask.shape)

        # Remove arm areas from the body mask
        final_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(arm_mask))

        # Refine the mask
        kernel = np.ones((5,5), np.uint8)
        final_mask = cv2.dilate(final_mask, kernel, iterations=2)
        final_mask = cv2.erode(final_mask, kernel, iterations=1)

        # Invert the mask for inpainting (white areas will be inpainted)
        inpaint_mask = cv2.bitwise_not(final_mask)

        return Image.fromarray(inpaint_mask)