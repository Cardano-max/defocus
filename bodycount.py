import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import mediapipe as mp
from scipy.spatial import Delaunay
from skimage.transform import PiecewiseAffineTransform, warp

class AdvancedGarmentFitter:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.segmentation_model = self.load_segmentation_model()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    def load_segmentation_model(self):
        model = deeplabv3_resnet101(pretrained=True)
        model.eval().to(self.device)
        return model

    def segment_garment(self, image):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.segmentation_model(input_tensor)['out'][0]
        mask = output.argmax(0).byte().cpu().numpy()
        return mask

    def get_garment_landmarks(self, image, mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get key points along the contour
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        landmarks = {}
        for i, point in enumerate(approx):
            x, y = point[0]
            landmarks[f'garment_{i}'] = [x, y]
        
        return landmarks

    def get_person_landmarks(self, image):
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            raise ValueError("No pose landmarks detected in the image.")
        
        landmarks = {}
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks[f'person_{i}'] = [landmark.x * image.shape[1], landmark.y * image.shape[0]]
        
        return landmarks

    def match_landmarks(self, garment_landmarks, person_landmarks):
        # This is a simplified matching. In a real scenario, you'd use a more sophisticated algorithm.
        matched_pairs = []
        for g_key, g_point in garment_landmarks.items():
            closest_p_key = min(person_landmarks, key=lambda p: np.linalg.norm(np.array(g_point) - np.array(person_landmarks[p])))
            matched_pairs.append((g_point, person_landmarks[closest_p_key]))
        return np.array(matched_pairs)

    def transform_garment(self, garment_image, person_image, matched_pairs):
        src_points = matched_pairs[:, 0]
        dst_points = matched_pairs[:, 1]
        
        # Create Delaunay triangulation
        tri = Delaunay(src_points)
        
        # Create piecewise affine transform
        transform = PiecewiseAffineTransform()
        transform.estimate(src_points, dst_points)
        
        # Apply the transform
        warped_garment = warp(garment_image, transform, output_shape=person_image.shape[:2])
        
        return (warped_garment * 255).astype(np.uint8)

    def blend_images(self, person_image, warped_garment):
        mask = cv2.threshold(cv2.cvtColor(warped_garment, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
        mask_inv = cv2.bitwise_not(mask)
        person_bg = cv2.bitwise_and(person_image, person_image, mask=mask_inv)
        garment_fg = cv2.bitwise_and(warped_garment, warped_garment, mask=mask)
        result = cv2.add(person_bg, garment_fg)
        return result

    def fit_garment(self, person_image_path, garment_image_path):
        person_image = cv2.imread(person_image_path)
        garment_image = cv2.imread(garment_image_path)
        
        if person_image is None or garment_image is None:
            raise ValueError("Failed to load images.")
        
        # Segment the garment
        garment_mask = self.segment_garment(Image.fromarray(cv2.cvtColor(garment_image, cv2.COLOR_BGR2RGB)))
        
        # Get landmarks
        garment_landmarks = self.get_garment_landmarks(garment_image, garment_mask)
        person_landmarks = self.get_person_landmarks(person_image)
        
        # Match landmarks
        matched_pairs = self.match_landmarks(garment_landmarks, person_landmarks)
        
        # Transform garment
        warped_garment = self.transform_garment(garment_image, person_image, matched_pairs)
        
        # Blend images
        result = self.blend_images(person_image, warped_garment)
        
        return result, garment_landmarks, person_landmarks

    def save_result(self, result, output_path):
        cv2.imwrite(output_path, result)
        print(f"Result saved to {output_path}")

    def visualize_landmarks(self, image, landmarks, output_path):
        for name, (x, y) in landmarks.items():
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(image, name, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        cv2.imwrite(output_path, image)
        print(f"Landmarks visualization saved to {output_path}")

# Usage
if __name__ == "__main__":
    fitter = AdvancedGarmentFitter()
    person_image_path = "TEST/mota.jpg"
    garment_image_path = "images/b9.png"
    output_path = "images/output_image.jpg"
    
    result, garment_landmarks, person_landmarks = fitter.fit_garment(person_image_path, garment_image_path)
    fitter.save_result(result, output_path)
    
    # Visualize landmarks
    person_image = cv2.imread(person_image_path)
    garment_image = cv2.imread(garment_image_path)
    fitter.visualize_landmarks(person_image.copy(), person_landmarks, "person_landmarks.jpg")
    fitter.visualize_landmarks(garment_image.copy(), garment_landmarks, "garment_landmarks.jpg")