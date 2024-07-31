import cv2
import numpy as np
from PIL import Image
from skimage import transform as tf
from skimage import measure
import mediapipe as mp
import os

class GarmentFitter:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)

    def detect_body_landmarks(self, image):
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            raise ValueError("No pose landmarks detected in the image.")
        return results.pose_landmarks

    def get_body_measurements(self, landmarks, image_shape):
        def dist(p1, p2):
            return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

        h, w = image_shape[:2]
        shoulder_width = dist(landmarks.landmark[11], landmarks.landmark[12]) * w
        chest_width = dist(landmarks.landmark[11], landmarks.landmark[12]) * 1.1 * w
        waist_width = dist(landmarks.landmark[23], landmarks.landmark[24]) * w
        hip_width = dist(landmarks.landmark[23], landmarks.landmark[24]) * 1.1 * w
        torso_height = (landmarks.landmark[23].y - landmarks.landmark[11].y) * h

        return {
            "shoulder_width": shoulder_width,
            "chest_width": chest_width,
            "waist_width": waist_width,
            "hip_width": hip_width,
            "torso_height": torso_height
        }

    def segment_garment(self, garment_image):
        gray = cv2.cvtColor(garment_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(garment_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
        return mask

    def get_garment_measurements(self, garment_mask):
        props = measure.regionprops(garment_mask)[0]
        return {
            "width": props.bbox[3] - props.bbox[1],
            "height": props.bbox[2] - props.bbox[0]
        }

    def create_transformation(self, garment_measurements, body_measurements):
        scale_x = body_measurements['chest_width'] / garment_measurements['width']
        scale_y = body_measurements['torso_height'] / garment_measurements['height']
        
        tform = tf.SimilarityTransform(scale=(scale_x, scale_y))
        return tform

    def apply_transformation(self, garment_image, tform, output_shape):
        return tf.warp(garment_image, tform, output_shape=output_shape, mode='constant', cval=0)

    def blend_images(self, background, foreground, mask):
        foreground = foreground.astype(np.float32) / 255.0
        background = background.astype(np.float32) / 255.0
        alpha = mask.astype(np.float32) / 255.0
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, background)
        output = cv2.add(foreground, background)
        return (output * 255).astype(np.uint8)

    def fit_garment(self, person_image, garment_image):
        landmarks = self.detect_body_landmarks(person_image)
        body_measurements = self.get_body_measurements(landmarks, person_image.shape)

        garment_mask = self.segment_garment(garment_image)
        garment_measurements = self.get_garment_measurements(garment_mask)

        tform = self.create_transformation(garment_measurements, body_measurements)
        transformed_garment = self.apply_transformation(garment_image, tform, person_image.shape)
        transformed_mask = self.apply_transformation(garment_mask, tform, person_image.shape)

        center_y = int((landmarks.landmark[11].y + landmarks.landmark[23].y) / 2 * person_image.shape[0])
        center_x = int((landmarks.landmark[11].x + landmarks.landmark[12].x) / 2 * person_image.shape[1])
        
        h, w = transformed_garment.shape[:2]
        top = max(0, center_y - h // 2)
        left = max(0, center_x - w // 2)
        
        placement_mask = np.zeros(person_image.shape[:2], dtype=np.uint8)
        placement_mask[top:top+h, left:left+w] = transformed_mask

        result = self.blend_images(person_image, transformed_garment, placement_mask)

        return result

def main():
    fitter = GarmentFitter()

    # Set up paths
    image_folder = "/Users/ikramali/projects/arbiosft_products/arbi-tryon/TEST"
    image_folder2 = "/Users/ikramali/projects/arbiosft_products/arbi-tryon/images"

    person_image_path = os.path.join(image_folder, "mota.jpg")
    garment_image_path = os.path.join(image_folder2, "b9.png")
    output_path = os.path.join(image_folder2, "fitted_garment_result.jpg")

    # Load images
    person_image = cv2.imread(person_image_path)
    garment_image = cv2.imread(garment_image_path)

    if person_image is None or garment_image is None:
        print("Error: Unable to load one or both images.")
        return

    # Fit garment
    result = fitter.fit_garment(person_image, garment_image)

    # Save result
    cv2.imwrite(output_path, result)
    print(f"Result saved to: {output_path}")

if __name__ == "__main__":
    main()