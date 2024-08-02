import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay
import torch
import torch.nn.functional as F

class GarmentFitter:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    def get_landmarks(self, image):
        results_pose = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results_face = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        landmarks = []
        if results_pose.pose_landmarks:
            for lm in results_pose.pose_landmarks.landmark:
                landmarks.append((int(lm.x * image.shape[1]), int(lm.y * image.shape[0])))
        
        if results_face.multi_face_landmarks:
            for lm in results_face.multi_face_landmarks[0].landmark:
                landmarks.append((int(lm.x * image.shape[1]), int(lm.y * image.shape[0])))
        
        return np.array(landmarks)

    def thin_plate_spline(self, source, target, regularization=0.0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        source = torch.tensor(source, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)

        n = source.shape[0]
        phi = torch.norm(source[:, None] - source[None, :], dim=2)
        phi = torch.where(phi == 0, torch.tensor(1e-9).to(device), phi)
        T = torch.log(phi)

        P = torch.ones((n, 3)).to(device)
        P[:, 1:] = source

        Z = torch.zeros((3, 3)).to(device)
        L = torch.cat([
            torch.cat([T, P], dim=1),
            torch.cat([P.t(), Z], dim=1)
        ], dim=0)

        Y = torch.cat([target, torch.zeros((3, 2)).to(device)], dim=0)

        L += torch.eye(n+3).to(device) * regularization
        param = torch.linalg.solve(L, Y).to(device)

        return param

    def apply_tps(self, image, source, target, param):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        height, width = image.shape[:2]
        grid = torch.meshgrid(torch.arange(height), torch.arange(width))
        grid = torch.stack(grid[::-1], dim=-1).float().to(device)

        source = torch.tensor(source, dtype=torch.float32).to(device)
        n = source.shape[0]

        diff = grid.view(-1, 1, 2) - source.view(1, n, 2)
        phi = torch.norm(diff, dim=2)
        phi = torch.where(phi == 0, torch.tensor(1e-9).to(device), phi)
        T = torch.log(phi)

        P = torch.ones((height*width, 3)).to(device)
        P[:, 1:] = grid.view(-1, 2)

        warped = torch.matmul(T, param[:n]) + torch.matmul(P, param[n:])
        warped = warped.view(height, width, 2)

        warped_image = F.grid_sample(torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0).to(device),
                                     warped.unsqueeze(0), align_corners=True)
        
        return warped_image.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    def fit_garment(self, person_image, garment_image):
        person_landmarks = self.get_landmarks(person_image)
        garment_landmarks = self.get_landmarks(garment_image)

        # Ensure we have the same number of landmarks for both images
        min_landmarks = min(len(person_landmarks), len(garment_landmarks))
        person_landmarks = person_landmarks[:min_landmarks]
        garment_landmarks = garment_landmarks[:min_landmarks]

        # Compute TPS parameters
        tps_param = self.thin_plate_spline(garment_landmarks, person_landmarks)

        # Apply TPS transformation
        warped_garment = self.apply_tps(garment_image, garment_landmarks, person_landmarks, tps_param)

        # Blend the warped garment with the person image
        mask = np.all(warped_garment != [0, 0, 0], axis=-1)
        result = np.where(mask[:, :, np.newaxis], warped_garment, person_image)

        return result

if __name__ == "__main__":
    fitter = GarmentFitter()
    person_image = cv2.imread('TEST/mota.jpg')
    garment_image = cv2.imread('images/b9.png', cv2.IMREAD_UNCHANGED)
    
    result = fitter.fit_garment(person_image, garment_image)
    cv2.imwrite('result.jpg', result)