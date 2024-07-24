import torch
from PIL import Image
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import os

# Initialize the RealESRGAN model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model_path = os.path.join(os.path.dirname(__file__), 'RealESRGAN_x4plus.pth')

if not os.path.exists(model_path):
    import gdown
    url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    gdown.download(url, model_path, quiet=False)

upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True
)

def restore_image(image):
    """
    Restore and enhance the input image using RealESRGAN.
    
    Args:
    image (PIL.Image or numpy.ndarray): Input image to be restored.
    
    Returns:
    numpy.ndarray: Restored image.
    """
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure the image is in RGB format
    if image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]
    
    # Process the image
    try:
        with torch.no_grad():
            output, _ = upsampler.enhance(image, outscale=2)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        # Convert back to uint8
        restored_image = output.astype(np.uint8)
        return restored_image

# Test the restoration function
if __name__ == "__main__":
    # Load a test image
    test_image_path = "/Users/ateeb.taseer/arbi_tryon/arbi-tryon/images/images.jpeg"
    test_image = Image.open(test_image_path)
    
    # Restore the image
    restored_image = restore_image(test_image)
    
    # Save the restored image
    Image.fromarray(restored_image).save("restored_test_image.png")
    print("Test image restored and saved as 'restored_test_image.png'")