import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

class GarmentFitter:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    def detect_body_landmarks(self, image):
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            raise ValueError("No body landmarks detected in the image.")
        return results.pose_landmarks

    def get_body_measurements(self, landmarks, image_shape):
        # Extract key measurements (simplified for this example)
        shoulder_left = np.array([landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_shape[1],
                                  landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_shape[0]])
        shoulder_right = np.array([landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_shape[1],
                                   landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_shape[0]])
        hip_left = np.array([landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x * image_shape[1],
                             landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y * image_shape[0]])
        hip_right = np.array([landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x * image_shape[1],
                              landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y * image_shape[0]])

        shoulder_width = np.linalg.norm(shoulder_right - shoulder_left)
        hip_width = np.linalg.norm(hip_right - hip_left)
        torso_height = np.mean([hip_left[1], hip_right[1]]) - np.mean([shoulder_left[1], shoulder_right[1]])

        return {
            "shoulder_width": shoulder_width,
            "hip_width": hip_width,
            "torso_height": torso_height,
            "shoulder_left": shoulder_left,
            "shoulder_right": shoulder_right,
            "hip_left": hip_left,
            "hip_right": hip_right
        }

    def detect_garment_keypoints(self, garment_image):
        # This is a placeholder. In a real implementation, you'd use a more sophisticated
        # method to detect garment keypoints, possibly using a trained model.
        gray = cv2.cvtColor(garment_image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        corners = np.int0(corners)
        return corners.reshape(-1, 2)

    def map_garment_to_body(self, person_image, garment_image, body_measurements, garment_keypoints):
        person_height, person_width = person_image.shape[:2]
        garment_height, garment_width = garment_image.shape[:2]

        # Calculate scaling factors
        scale_x = body_measurements["shoulder_width"] / garment_width
        scale_y = body_measurements["torso_height"] / garment_height

        # Create transformation matrix
        src_points = np.float32([
            [0, 0],
            [garment_width - 1, 0],
            [0, garment_height - 1],
            [garment_width - 1, garment_height - 1]
        ])

        dst_points = np.float32([
            body_measurements["shoulder_left"],
            body_measurements["shoulder_right"],
            body_measurements["hip_left"],
            body_measurements["hip_right"]
        ])

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the transformation
        warped_garment = cv2.warpPerspective(garment_image, matrix, (person_width, person_height))

        return warped_garment

    def blend_images(self, person_image, warped_garment):
        # Create a mask for the warped garment
        mask = cv2.cvtColor(warped_garment, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Invert the mask
        mask_inv = cv2.bitwise_not(mask)

        # Black-out the area of the garment in the person image
        person_bg = cv2.bitwise_and(person_image, person_image, mask=mask_inv)

        # Take only the region of the garment from the warped image
        garment_fg = cv2.bitwise_and(warped_garment, warped_garment, mask=mask)

        # Combine the two images
        result = cv2.add(person_bg, garment_fg)

        return result

    def fit_garment(self, person_image_path, garment_image_path):
        person_image = cv2.imread(person_image_path)
        garment_image = cv2.imread(garment_image_path)

        if person_image is None or garment_image is None:
            raise ValueError("Unable to read one or both of the input images.")

        landmarks = self.detect_body_landmarks(person_image)
        body_measurements = self.get_body_measurements(landmarks, person_image.shape)
        garment_keypoints = self.detect_garment_keypoints(garment_image)

        warped_garment = self.map_garment_to_body(person_image, garment_image, body_measurements, garment_keypoints)
        result = self.blend_images(person_image, warped_garment)

        return result

    def visualize_keypoints(self, image, keypoints):
        for point in keypoints:
            x, y = point.ravel()
            cv2.circle(image, (int(x), int(y)), 3, 255, -1)
        return image

if __name__ == "__main__":
    fitter = GarmentFitter()
    
    person_image_path = "TEST/mota.jpg"
    garment_image_path = "images/b9.png"
    
    try:
        result = fitter.fit_garment(person_image_path, garment_image_path)
        cv2.imwrite("fitted_garment.jpg", result)
        print("Garment fitting completed. Result saved as 'fitted_garment.jpg'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")