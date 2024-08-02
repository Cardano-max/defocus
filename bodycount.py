import cv2
import numpy as np

def tps_warp(src_image, src_points, dst_points):
    tps = cv2.createThinPlateSplineShapeTransformer()

    src_points = src_points.astype(np.float32).reshape(1, -1, 2)
    dst_points = dst_points.astype(np.float32).reshape(1, -1, 2)

    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points[0]))]

    tps.estimateTransformation(dst_points, src_points, matches)
    warped_image = tps.warpImage(src_image)

    return warped_image

def main():
    # Load the image
    image_path = 'images/b9.png'  # Replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Failed to load image from {image_path}")
        return

    # Get image dimensions
    height, width = image.shape[:2]

    # Define source points (corners of the image)
    src_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
        [width // 2, height // 2]  # Center point
    ])

    # Define destination points (where you want the corners to move)
    # This example will "pinch" the image towards the center
    dst_points = np.array([
        [width * 0.1, height * 0.1],  # Top-left moved inwards
        [width * 0.9, height * 0.1],  # Top-right moved inwards
        [width * 0.9, height * 0.9],  # Bottom-right moved inwards
        [width * 0.1, height * 0.9],  # Bottom-left moved inwards
        [width // 2, height // 2]  # Center point (unchanged)
    ])

    # Apply TPS warping
    warped_image = tps_warp(image, src_points, dst_points)

    # Save the result
    cv2.imwrite('warped_image.png', warped_image)
    print("Warped image saved as 'warped_image.png'")

    # Display the original and warped images
    cv2.imshow('Original Image', image)
    cv2.imshow('Warped Image', warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()