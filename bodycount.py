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
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
        self.api_token = os.environ.get("r8_IeCOnHH0kGL7jscpqy4r7LLggru0ELM0d1qDU")
        if not self.api_token:
            raise ValueError("Please set the REPLICATE_API_TOKEN environment variable.")

    def get_landmarks(self, image, is_garment=False):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if is_garment:
            results = self.hands.process(rgb_image)
            if not results.multi_hand_landmarks:
                raise ValueError("No landmarks detected on the garment.")
            landmarks = results.multi_hand_landmarks[0].landmark
        else:
            results = self.pose.process(rgb_image)
            if not results.pose_landmarks:
                raise ValueError("No pose landmarks detected on the person.")
            landmarks = results.pose_landmarks.landmark

        h, w, _ = image.shape
        return {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}

    def map_landmarks(self, garment_landmarks, person_landmarks):
        # Map key points between garment and person
        mapping = {
            0: self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            1: self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            2: self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            3: self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            4: self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            5: self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
        }
        return {garment_key: person_landmarks[person_key] for garment_key, person_key in mapping.items()}

    def run_draggan(self, input_image, source_coords, target_coords):
        model = "zsxkib/draggan:196f5c18c936529b12dac7fcce3c72e40a73b440d88ebdba8bb1bf0cf8944525"
        
        # Convert coordinates to the format expected by the API
        source_coords_str = f"({source_coords[1]}, {source_coords[0]})"
        target_coords_str = f"({target_coords[1]}, {target_coords[0]})"

        input_data = {
            "learning_rate": 0.004,
            "stylegan2_model": "self_distill/parrots_512_pytorch.pkl",  # You may need to change this based on your use case
            "source_pixel_coords": source_coords_str,
            "target_pixel_coords": target_coords_str,
            "maximum_n_iterations": 50,
            "show_points_and_arrows": True,
            "only_render_first_frame": False
        }

        # Run the model
        output = replicate.run(model, input=input_data)
        
        return output

    def download_image(self, url, output_path):
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content))
        img.save(output_path)

    def fit_garment(self, garment_image_path, person_image_path, output_dir):
        garment_image = cv2.imread(garment_image_path)
        person_image = cv2.imread(person_image_path)

        garment_landmarks = self.get_landmarks(garment_image, is_garment=True)
        person_landmarks = self.get_landmarks(person_image)

        person_landmarks_mapped = self.map_landmarks(garment_landmarks, person_landmarks)

        # Process each landmark pair
        for garment_key, garment_point in garment_landmarks.items():
            if garment_key in person_landmarks_mapped:
                person_point = person_landmarks_mapped[garment_key]
                
                # Run DragGAN for each point
                output_url = self.run_draggan(garment_image_path, garment_point, person_point)
                
                # Download and save the result
                output_path = os.path.join(output_dir, f"fitted_garment_{garment_key}.png")
                self.download_image(output_url, output_path)
                
                print(f"Processed point {garment_key}. Result saved to {output_path}")

        print(f"All points processed. Results saved in {output_dir}")

if __name__ == "__main__":
    fitter = GarmentFitter()
    garment_image_path = "images/b9.png"
    person_image_path = "TEST/mota.jpg"
    output_dir = "images/"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    fitter.fit_garment(garment_image_path, person_image_path, output_dir)