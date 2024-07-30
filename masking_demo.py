import numpy as np
import cv2
import os
from PIL import Image
from Masking.preprocess.humanparsing.run_parsing import Parsing
from Masking.preprocess.openpose.run_openpose import OpenPose
import paramiko
import io
import traceback


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

    def get_mask(self, img, category='upper_body'):
        # Resize image to 384x512 for processing
        img_resized = img.resize((384, 512), Image.Resampling.LANCZOS)
        
        # Get human parsing result
        parse_result, _ = self.parsing_model(img_resized)
        parse_array = np.array(parse_result)

        # Get pose estimation
        keypoints = self.openpose_model(img_resized)
        pose_data = np.array(keypoints["pose_keypoints_2d"]).reshape((-1, 2))

        # Create initial mask based on category
        if category == 'upper_body':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"]])
        elif category == 'lower_body':
            mask = np.isin(parse_array, [self.label_map["pants"], self.label_map["skirt"]])
        elif category == 'dresses':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"], 
                                         self.label_map["pants"], self.label_map["skirt"]])
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

        # Create arm mask
        arm_mask = np.isin(parse_array, [self.label_map["left_arm"], self.label_map["right_arm"]])

        # Create hand mask using pose data
        hand_mask = self.create_hand_mask(pose_data, parse_array.shape)

        # Combine arm and hand mask
        arm_hand_mask = np.logical_or(arm_mask, hand_mask)

        # Remove arms and hands from the mask
        mask = np.logical_and(mask, np.logical_not(arm_hand_mask))

        # Refine the mask
        mask = self.refine_mask(mask)

        # Resize mask back to original image size
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize(img.size, Image.Resampling.LANCZOS)
        
        return np.array(mask_pil)

    def create_hand_mask(self, pose_data, shape):
        hand_mask = np.zeros(shape, dtype=np.uint8)
        
        # Right hand
        if pose_data[4][0] > 0 and pose_data[4][1] > 0:  # If right wrist is detected
            cv2.circle(hand_mask, (int(pose_data[4][0]), int(pose_data[4][1])), 30, 255, -1)
        
        # Left hand
        if pose_data[7][0] > 0 and pose_data[7][1] > 0:  # If left wrist is detected
            cv2.circle(hand_mask, (int(pose_data[7][0]), int(pose_data[7][1])), 30, 255, -1)
        
        return hand_mask > 0  # Convert back to boolean mask

    def refine_mask(self, mask):
        # Convert to uint8 for OpenCV operations
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Apply morphological operations to smooth the mask
        kernel = np.ones((5,5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # Find contours and keep only the largest one
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_refined = np.zeros_like(mask_uint8)
            cv2.drawContours(mask_refined, [largest_contour], 0, 255, -1)
        else:
            mask_refined = mask_uint8
        
        return mask_refined > 0  # Convert back to boolean mask

def transfer_file(local_path, remote_path, sftp):
    try:
        sftp.put(local_path, remote_path)
        print(f"File transferred successfully to {remote_path}")
    except Exception as e:
        print(f"Error transferring file: {str(e)}")

if __name__ == "__main__":
    # SSH connection details
    hostname = "172.30.1.80"
    username = "ikramali"
    password = "arbisoft042"

    # Remote and local paths
    remote_image_path = "arbi-tryon/TEST/mota.jpg"
    local_output_dir = "/Users/ikramali/projects/arbiosft_products/arbi-tryon/TEST/"

    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=username, password=password)

    # Create SFTP client
    sftp = ssh.open_sftp()

    try:
        # Download the image from remote to local memory
        with sftp.open(remote_image_path, 'rb') as remote_file:
            image_data = remote_file.read()
        
        # Process the image
        human_img = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        masker = Masking()
        mask = masker.get_mask(human_img, category='upper_body')
        
        # Convert the mask to OpenCV format
        mask_cv2 = cv2.cvtColor(np.array(mask, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Create a masked image by applying the mask to the original image
        masked_image = cv2.bitwise_and(cv2.cvtColor(np.array(human_img), cv2.COLOR_RGB2BGR), mask_cv2)
        
        # Save the mask and masked image locally
        cv2.imwrite(os.path.join(local_output_dir, "output_mask.png"), mask_cv2)
        cv2.imwrite(os.path.join(local_output_dir, "output_masked_image.png"), masked_image)
        
        print(f"Mask and masked image saved in {local_output_dir}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()

    finally:
        # Close the SFTP and SSH connections
        sftp.close()
        ssh.close()