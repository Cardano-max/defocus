import numpy as np
import torch
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import mediapipe as mp

class Masking:
    def __init__(self):
        self.processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        self.label_map = {
            "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
            "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
            "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
            "bag": 16, "scarf": 17
        }

    def detect_body_parts(self, image):
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            return np.array([[lm.x * image.shape[1], lm.y * image.shape[0]] for lm in landmarks])
        return None

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

    def refine_mask(self, mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refined_mask = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Adjust this threshold as needed
                cv2.drawContours(refined_mask, [contour], 0, 255, -1)
        return refined_mask

    def get_mask(self, image, category='upper_body'):
        # Ensure image is in RGB mode and convert to numpy array
        image_np = np.array(image.convert('RGB'))
        height, width = image_np.shape[:2]

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

        # Detect body parts and hands
        body_landmarks = self.detect_body_parts(image_np)
        hand_masks = self.detect_hands(image_np)

        # Create initial mask based on category
        if category == 'upper_body':
            initial_mask = ((parse_array == self.label_map["upper_clothes"]) |
                            (parse_array == self.label_map["dress"])).astype(np.uint8) * 255
        elif category == 'lower_body':
            initial_mask = ((parse_array == self.label_map["pants"]) |
                            (parse_array == self.label_map["skirt"])).astype(np.uint8) * 255
        elif category == 'dresses':
            initial_mask = (parse_array == self.label_map["dress"]).astype(np.uint8) * 255
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

        # Refine the initial mask
        refined_mask = self.refine_mask(initial_mask)

        # Create a separate mask for hands
        hand_mask = np.zeros_like(refined_mask)
        for mask in hand_masks:
            hand_mask = cv2.bitwise_or(hand_mask, mask)

        # Dilate the refined mask
        kernel = np.ones((5,5), np.uint8)
        dilated_mask = cv2.dilate(refined_mask, kernel, iterations=3)

        # Remove hand areas from the dilated mask
        final_mask = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(hand_mask))

        # Create segmentation image
        segmentation = np.zeros((height, width, 3), dtype=np.uint8)
        for label, color in zip(self.label_map.values(), np.random.randint(0, 255, (len(self.label_map), 3))):
            segmentation[parse_array == label] = color

        return Image.fromarray(segmentation), Image.fromarray(final_mask)

def process_image(image_path, category='upper_body'):
    masker = Masking()
    image = Image.open(image_path)
    segmentation, mask = masker.get_mask(image, category)
    
    # Save outputs
    segmentation.save('segmentation_output.png')
    mask.save('mask_output.png')
    
    return segmentation, mask

# Example usage
if __name__ == "__main__":
    image_path = "arbi-tryon/images/ok.png"
    category = "upper_body"  # or "lower_body" or "dresses"
    segmentation, mask = process_image(image_path, category)
    print("Segmentation and mask images have been saved.")