import numpy as np
from PIL import Image
from pathlib import Path
from functools import wraps
from time import time
import torch

# Ensure torch uses CPU if CUDA is not available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import SegBody (assuming it's in the same directory)
from SegBody import segment_body

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} args:[{args}, {kw}] took: {te-ts:.4f} sec')
        return result
    return wrap

class Masking:
    def __init__(self):
        pass

    @timing
    def get_mask(self, img, category='full_body'):
        # Resize image to 512x512
        img_resized = img.resize((512, 512), Image.LANCZOS)
        
        # Use SegBody to create the mask
        _, mask_image = segment_body(img_resized, face=False)
        
        # Convert mask to binary (0 or 255)
        mask_array = np.array(mask_image)
        binary_mask = (mask_array > 128).astype(np.uint8) * 255
        
        # Create PIL Image from numpy array
        mask_pil = Image.fromarray(binary_mask)
        
        # Resize mask back to original image size
        mask_pil = mask_pil.resize(img.size, Image.LANCZOS)
        
        return mask_pil

def process_images(input_folder, output_folder, category='full_body'):
    masker = Masking()
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    image_files = list(Path(input_folder).glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')]
    
    for i, image_file in enumerate(image_files, 1):
        output_mask = Path(output_folder) / f"output_mask_{i}.png"
        output_masked = Path(output_folder) / f"output_masked_image_{i}.png"
        
        print(f"Processing image {i}/{len(image_files)}: {image_file.name}")
        
        input_img = Image.open(image_file).convert('RGB')
        
        mask = masker.get_mask(input_img, category=category)
        
        # Save the mask
        mask.save(str(output_mask))
        
        # Apply the mask to the original image
        masked_output = Image.composite(input_img, Image.new('RGB', input_img.size, (255, 255, 255)), mask)
        masked_output.save(str(output_masked))
        
        print(f"Mask saved to {output_mask}")
        print(f"Masked output saved to {output_masked}")
        print()

if __name__ == "__main__":
    input_folder = Path("/path/to/input/folder")
    output_folder = Path("/path/to/output/folder")
    category = "full_body"  # This parameter is kept for consistency, but not used in the SegBody method
    
    process_images(str(input_folder), str(output_folder), category)


if __name__ == "__main__":
    input_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/in_im")
    output_folder = Path("/Users/ikramali/projects/arbiosft_products/arbi-tryon/output")
    category = "dresses"  # Change to "upper_body", "lower_body", or "dresses" as needed
    
    process_images(str(input_folder), str(output_folder), category)