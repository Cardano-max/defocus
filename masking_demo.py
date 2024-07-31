import numpy as np
import cv2
from PIL import Image
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

class Masking:
    def __init__(self):
        self.label_map = {
            "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
            "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
            "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
            "bag": 16, "scarf": 17, "neck": 18
        }

    @timing
    def get_mask(self, img, category='upper_body'):
        # Convert PIL Image to numpy array
        img_np = np.array(img)
        
        # Create initial mask based on category
        mask = self.create_initial_mask(img_np, category)
        
        # Apply smooth transition to edges
        smooth_mask = self.apply_smooth_transition(mask)
        
        # Create masked output image
        masked_output = self.create_masked_output(img_np, smooth_mask)
        
        return smooth_mask, masked_output

    def create_initial_mask(self, image, category):
        # Simulating a segmentation result for demonstration
        # In a real scenario, you would use your existing segmentation model here
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if category == 'upper_body':
            mask[height//4:3*height//4, width//4:3*width//4] = 255
        elif category == 'lower_body':
            mask[height//2:, width//4:3*width//4] = 255
        elif category == 'dresses':
            mask[height//4:, width//4:3*width//4] = 255
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")
        
        return mask

    def apply_smooth_transition(self, mask, blur_radius=31, transition_width=20):
        # Ensure blur_radius is odd
        blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        
        # Create a copy of the mask
        smooth_mask = mask.copy().astype(np.float32)
        
        # Apply Gaussian blur to create initial smoothing
        blurred = cv2.GaussianBlur(smooth_mask, (blur_radius, blur_radius), 0)
        
        # Create a transition mask
        kernel = np.ones((transition_width, transition_width), np.uint8)
        dilated = cv2.dilate(smooth_mask, kernel, iterations=1)
        eroded = cv2.erode(smooth_mask, kernel, iterations=1)
        transition_mask = dilated - eroded
        
        # Apply the transition
        smooth_mask = np.where(transition_mask > 0, blurred, smooth_mask)
        
        # Normalize to 0-255 range
        smooth_mask = cv2.normalize(smooth_mask, None, 0, 255, cv2.NORM_MINMAX)
        
        return smooth_mask.astype(np.uint8)

    def create_masked_output(self, image, mask):
        # Ensure mask is binary
        binary_mask = (mask > 127).astype(np.uint8) * 255
        
        # Apply the mask to the original image
        masked_output = cv2.bitwise_and(image, image, mask=binary_mask)
        
        return masked_output

if __name__ == "__main__":
    import os
    
    masker = Masking()
    image_folder = "/Users/ikramali/projects/arbiosft_products/arbi-tryon/TEST"
    input_image = os.path.join(image_folder, "mota2.png")
    output_mask = os.path.join(image_folder, "output_smooth_mask.png")
    output_masked = os.path.join(image_folder, "output_masked_image.png")
    category = "dresses"  # Change this to "lower_body" or "dresses" as needed
    
    # Load the input image
    input_img = Image.open(input_image)
    
    # Get the smooth mask and masked output
    smooth_mask, masked_output = masker.get_mask(input_img, category=category)
    
    # Save the output smooth mask image
    Image.fromarray(smooth_mask).save(output_mask)
    
    # Save the output masked image
    Image.fromarray(masked_output).save(output_masked)
    
    print(f"Smooth mask saved to {output_mask}")
    print(f"Masked output saved to {output_masked}")