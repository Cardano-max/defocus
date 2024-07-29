import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from Masking.preprocess.openpose.run_openpose import OpenPose
import mediapipe as mp
import matplotlib.pyplot as plt

class Masking:
    def __init__(self):
        self.processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")
        self.openpose = OpenPose(gpu_id=0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        self.label_map = {
            "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
            "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
            "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
            "bag": 16, "scarf": 17
        }

    def extend_arm_mask(self, wrist, elbow, scale):
        return elbow + scale * (wrist - elbow)

    def hole_fill(self, img):
        img = np.pad(img[1:-1, 1:-1], pad_width=1, mode='constant', constant_values=0)
        img_copy = img.copy()
        mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(img, mask, (0, 0), 255)
        img_inverse = cv2.bitwise_not(img)
        return cv2.bitwise_or(img_copy, img_inverse)

    def refine_mask(self, mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        area = [abs(cv2.contourArea(contour, True)) for contour in contours]
        if area:
            max_idx = area.index(max(area))
            refined_mask = np.zeros_like(mask).astype(np.uint8)
            cv2.drawContours(refined_mask, contours, max_idx, color=255, thickness=-1)
            return refined_mask
        return mask

    def detect_hands(self, image):
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        hand_masks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                landmarks = [[int(lm.x * image.shape[1]), int(lm.y * image.shape[0])] for lm in hand_landmarks.landmark]
                hull = cv2.convexHull(np.array(landmarks))
                cv2.fillConvexPoly(hand_mask, hull, 255)
                hand_masks.append(hand_mask)
        return hand_masks

    def get_mask(self, image, category='upper_body', width=384, height=512):
        # Ensure image is in RGB mode
        image = image.convert('RGB')
        
        # Resize image
        image = image.resize((width, height), Image.NEAREST)
        
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

        # Visualize segmentation
        segmentation_image = np.zeros((height, width, 3), dtype=np.uint8)
        for label, color in self.label_map.items():
            segmentation_image[parse_array == color] = [color * 15, color * 15, color * 15]

        # Get pose estimation
        keypoints = self.openpose(np.array(image))
        pose_data = np.array(keypoints["pose_keypoints_2d"]).reshape((-1, 2))

        # Detect hands
        hand_masks = self.detect_hands(np.array(image))

        # Create masks
        parse_head = ((parse_array == 1) | (parse_array == 3) | (parse_array == 11)).astype(np.float32)
        parser_mask_fixed = ((parse_array == self.label_map["left_shoe"]) |
                             (parse_array == self.label_map["right_shoe"]) |
                             (parse_array == self.label_map["hat"]) |
                             (parse_array == self.label_map["sunglasses"]) |
                             (parse_array == self.label_map["bag"])).astype(np.float32)
        parser_mask_changeable = (parse_array == self.label_map["background"]).astype(np.float32)

        if category == 'upper_body':
            parse_mask = ((parse_array == 4) | (parse_array == 7)).astype(np.float32)
            parser_mask_fixed_lower = ((parse_array == self.label_map["skirt"]) |
                                       (parse_array == self.label_map["pants"])).astype(np.float32)
            parser_mask_fixed += parser_mask_fixed_lower
            parser_mask_changeable |= ~(parser_mask_fixed | parse_mask)
        elif category == 'lower_body':
            parse_mask = ((parse_array == 6) | (parse_array == 12) |
                          (parse_array == 13) | (parse_array == 5)).astype(np.float32)
            parser_mask_fixed += ((parse_array == self.label_map["upper_clothes"]) |
                                  (parse_array == 14) | (parse_array == 15)).astype(np.float32)
            parser_mask_changeable |= ~(parser_mask_fixed | parse_mask)
        elif category == 'dresses':
            parse_mask = ((parse_array == 7) | (parse_array == 4) |
                          (parse_array == 5) | (parse_array == 6)).astype(np.float32)
            parser_mask_changeable |= ~(parser_mask_fixed | parse_mask)
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

        # Process arms
        arm_width = int(0.15 * width)  # Adjust arm width based on image width
        im_arms = Image.new('L', (width, height))
        arms_draw = ImageDraw.Draw(im_arms)

        if category in ['dresses', 'upper_body']:
            for side in ['right', 'left']:
                shoulder = pose_data[2 if side == 'right' else 5]
                elbow = pose_data[3 if side == 'right' else 6]
                wrist = pose_data[4 if side == 'right' else 7]

                if wrist[0] > 1 and wrist[1] > 1:
                    wrist = self.extend_arm_mask(wrist, elbow, 1.2)
                    arms_draw.line(np.concatenate((shoulder, elbow, wrist)).astype(np.uint16).tolist(),
                                   fill='white', width=arm_width, joint='curve')
                    arms_draw.ellipse([shoulder[0] - arm_width // 2, shoulder[1] - arm_width // 2,
                                       shoulder[0] + arm_width // 2, shoulder[1] + arm_width // 2],
                                      fill='white')

        arm_mask = np.array(im_arms) / 255.0
        parse_mask = np.logical_or(parse_mask, arm_mask).astype(np.float32)

        # Unmask hands before dilation
        for hand_mask in hand_masks:
            parse_mask[hand_mask > 0] = 0

        # Final mask processing
        parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
        neck_mask = (parse_array == 18).astype(np.float32)
        neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
        parse_mask = np.logical_or(parse_mask, neck_mask)

        inpaint_mask = 1 - (parser_mask_changeable | parse_mask | parser_mask_fixed)
        inpaint_mask = self.hole_fill((inpaint_mask * 255).astype(np.uint8))
        inpaint_mask = self.refine_mask(inpaint_mask)

        # Unmask hands again after refinement
        for hand_mask in hand_masks:
            inpaint_mask[hand_mask > 0] = 0

        # Visualize final mask
        mask_image = inpaint_mask.copy()
        
        return Image.fromarray(inpaint_mask), Image.fromarray(segmentation_image), Image.fromarray(mask_image)