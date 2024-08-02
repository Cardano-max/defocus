import cv2
import numpy as np
import mediapipe as mp
import replicate
import os
import requests
from PIL import Image
import io

class GarmentFitter:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not self.api_token:
            raise ValueError("Please set the REPLICATE_API_TOKEN environment variable.")
        os.environ["REPLICATE_API_TOKEN"] = self.api_token

    def get_person_landmarks(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        if not results.pose_landmarks:
            raise ValueError("No pose landmarks detected on the person.")
        h, w, _ = image.shape
        return {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(results.pose_landmarks.landmark)}

    def get_garment_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        if corners is None:
            raise ValueError("No corners detected on the garment.")
        return np.int0(corners).reshape(-1, 2)

    def map_landmarks(self, garment_landmarks, person_landmarks):
        # Define key points for mapping
        garment_points = [
            garment_landmarks[0],  # Top left
            garment_landmarks[-1],  # Bottom right
            garment_landmarks[len(garment_landmarks)//2],  # Center
        ]
        person_points = [
            person_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            person_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            person_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
        ]
        return list(zip(garment_points, person_points))

    def apply_perspective_transform(self, src_img, src_points, dst_points):
        src_points = np.float32(src_points)
        dst_points = np.float32(dst_points)
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        result = cv2.warpPerspective(src_img, matrix, (src_img.shape[1], src_img.shape[0]))
        return result

    def run_draggan(self, input_image, source_coords, target_coords):
        model = "zsxkib/draggan:196f5c18c936529b12dac7fcce3c72e40a73b440d88ebdba8bb1bf0cf8944525"
        
        input_data = {
            "image": input_image,
            "learning_rate": 0.004,
            "stylegan2_model": "self_distill/parrots_512_pytorch.pkl",
            "source_pixel_coords": f"({source_coords[1]}, {source_coords[0]})",
            "target_pixel_coords": f"({target_coords[1]}, {target_coords[0]})",
            "maximum_n_iterations": 50,
            "show_points_and_arrows": True,
        }

        output = replicate.run(model, input=input_data)
        return output

    def download_image(self, url):
        response = requests.get(url)
        return Image.open(io.BytesIO(response.content))

    def fit_garment(self, garment_image_path, person_image_path, output_path):
        garment_image = cv2.imread(garment_image_path)
        person_image = cv2.imread(person_image_path)

        garment_landmarks = self.get_garment_landmarks(garment_image)
        person_landmarks = self.get_person_landmarks(person_image)

        mapped_landmarks = self.map_landmarks(garment_landmarks, person_landmarks)

        # Apply perspective transform
        transformed_garment = self.apply_perspective_transform(
            garment_image,
            [point[0] for point in mapped_landmarks],
            [point[1] for point in mapped_landmarks]
        )

        # Fine-tune with DragGAN
        for src, dst in mapped_landmarks:
            output_url = self.run_draggan(transformed_garment, src, dst)
            transformed_garment = np.array(self.download_image(output_url))

        # Blend the transformed garment with the person image
        mask = np.zeros(person_image.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        rect = (50, 50, person_image.shape[1]-100, person_image.shape[0]-100)
        cv2.grabCut(transformed_garment, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        transformed_garment = transformed_garment*mask2[:,:,np.newaxis]

        result = cv2.addWeighted(person_image, 0.7, transformed_garment, 0.3, 0)

        cv2.imwrite(output_path, result)
        print(f"Fitted garment saved to {output_path}")

if __name__ == "__main__":
    fitter = GarmentFitter()
    garment_image_path = "images/b9.png"
    person_image_path = "TEST/mota.jpg"
    output_path = "images/"

    fitter.fit_garment(garment_image_path, person_image_path, output_path)