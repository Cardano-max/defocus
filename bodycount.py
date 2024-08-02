import cv2
import numpy as np
import mediapipe as mp

def get_face_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        return np.array([(lm.x * image.shape[1], lm.y * image.shape[0]) for lm in results.multi_face_landmarks[0].landmark])

def get_body_landmarks(image):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        return np.array([(lm.x * image.shape[1], lm.y * image.shape[0]) for lm in results.pose_landmarks.landmark])

def WarpImage_TPS(source, target, img):
    tps = cv2.createThinPlateSplineShapeTransformer()

    source = source.reshape(-1, len(source), 2)
    target = target.reshape(-1, len(target), 2)

    matches = [cv2.DMatch(i, i, 0) for i in range(len(source[0]))]

    tps.estimateTransformation(target, source, matches)
    new_img = tps.warpImage(img)

    return new_img

def fit_garment(person_image, garment_image):
    # Get person landmarks
    face_landmarks = get_face_landmarks(person_image)
    body_landmarks = get_body_landmarks(person_image)
    
    if face_landmarks is None or body_landmarks is None:
        raise ValueError("Could not detect face or body landmarks")

    # Define key points for garment fitting
    left_shoulder = body_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = body_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = body_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    right_hip = body_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

    # Define source points on the garment
    garment_height, garment_width = garment_image.shape[:2]
    source_points = np.array([
        [0, 0],
        [garment_width, 0],
        [0, garment_height],
        [garment_width, garment_height],
        [garment_width/2, 0],
        [garment_width/2, garment_height],
        [0, garment_height/2],
        [garment_width, garment_height/2]
    ])

    # Define target points on the person
    target_points = np.array([
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip,
        (left_shoulder + right_shoulder) / 2,
        (left_hip + right_hip) / 2,
        (left_shoulder + left_hip) / 2,
        (right_shoulder + right_hip) / 2
    ])

    # Warp the garment image
    warped_garment = WarpImage_TPS(source_points, target_points, garment_image)

    # Blend the warped garment with the person image
    mask = cv2.threshold(cv2.cvtColor(warped_garment, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
    mask_inv = cv2.bitwise_not(mask)
    person_bg = cv2.bitwise_and(person_image, person_image, mask=mask_inv)
    garment_fg = cv2.bitwise_and(warped_garment, warped_garment, mask=mask)
    result = cv2.add(person_bg, garment_fg)

    return result, warped_garment

# Main execution
if __name__ == "__main__":
    # Load images
    person_image = cv2.imread('TEST/mota.jpg')
    garment_image = cv2.imread('images/b9.png', cv2.IMREAD_UNCHANGED)

    # Fit garment
    try:
        result, warped_garment = fit_garment(person_image, garment_image)
        
        # Save results
        cv2.imwrite('fitted_garment.jpg', result)
        cv2.imwrite('warped_garment.png', warped_garment)
        
        print("Processing completed. Check 'fitted_garment.jpg' and 'warped_garment.png' for results.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")