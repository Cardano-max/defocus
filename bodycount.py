import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

class GarmentFitter:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    def get_body_landmarks(self, image):
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            raise ValueError("No pose landmarks detected in the image.")
        return np.array([[lm.x * image.shape[1], lm.y * image.shape[0]] for lm in results.pose_landmarks.landmark])

    def get_garment_landmarks(self, garment_image):
        # Simplified garment landmark detection
        # In a real implementation, you'd use a more sophisticated method
        gray = cv2.cvtColor(garment_image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        return np.int0(corners).reshape(-1, 2)

    def map_coordinates(self, person_landmarks, garment_landmarks, person_image, garment_image):
        # Define key points for mapping (simplified)
        person_points = np.float32([
            person_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            person_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            person_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            person_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        ])

        garment_height, garment_width = garment_image.shape[:2]
        garment_points = np.float32([
            [0, 0],
            [garment_width - 1, 0],
            [0, garment_height - 1],
            [garment_width - 1, garment_height - 1]
        ])

        # Calculate the transformation matrix
        M = cv2.getPerspectiveTransform(garment_points, person_points)

        # Apply the transformation to the garment image
        person_height, person_width = person_image.shape[:2]
        warped_garment = cv2.warpPerspective(garment_image, M, (person_width, person_height))

        return warped_garment

    def blend_images(self, person_image, warped_garment):
        # Create a mask for the warped garment
        mask = cv2.threshold(cv2.cvtColor(warped_garment, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]

        # Invert the mask
        mask_inv = cv2.bitwise_not(mask)

        # Black-out the area of the garment in the person image
        person_bg = cv2.bitwise_and(person_image, person_image, mask=mask_inv)

        # Take only the region of the garment from the warped image
        garment_fg = cv2.bitwise_and(warped_garment, warped_garment, mask=mask)

        # Combine the person and garment
        result = cv2.add(person_bg, garment_fg)

        return result

    def fit_garment(self, person_image_path, garment_image_path):
        person_image = cv2.imread(person_image_path)
        garment_image = cv2.imread(garment_image_path)

        person_landmarks = self.get_body_landmarks(person_image)
        garment_landmarks = self.get_garment_landmarks(garment_image)

        warped_garment = self.map_coordinates(person_landmarks, garment_landmarks, person_image, garment_image)
        result = self.blend_images(person_image, warped_garment)

        return result

    def save_result(self, result, output_path):
        cv2.imwrite(output_path, result)
        print(f"Result saved to {output_path}")

# Usage
if __name__ == "__main__":
    fitter = GarmentFitter()
    result = fitter.fit_garment("TEST/mota.jpg", "images/b9.png")
    fitter.save_result(result, "/Users/ateeb.taseer/arbi_tryon/arbi-tryon/images")