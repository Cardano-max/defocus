import os
import time
import traceback
import numpy as np
import torch
from PIL import Image
import cv2
from queue import Queue
from threading import Lock, Event
import modules.config
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
from modules.util import HWC3, resize_image
from Masking.masking import Masking

# Initialize Masker
masker = Masking()

# Function to generate mask
def generate_mask(person_image, category="dresses"):
    if not isinstance(person_image, Image.Image):
        person_image = Image.fromarray(person_image)
    
    inpaint_mask = masker.get_mask(person_image, category=category)
    return inpaint_mask

# Main function for virtual try-on
def virtual_try_on(person_image_path, prompt, category="dresses", output_path=None):
    try:
        # Load person image
        person_image = Image.open(person_image_path)
        person_image = np.array(person_image)

        # Generate mask
        inpaint_mask = generate_mask(person_image, category)

        # Get the original dimensions of the person image
        orig_person_h, orig_person_w = person_image.shape[:2]

        # Calculate the aspect ratio of the person image
        person_aspect_ratio = orig_person_h / orig_person_w

        # Set target width and calculate corresponding height to maintain aspect ratio
        target_width = 1024
        target_height = int(target_width * person_aspect_ratio)

        # Ensure target height is also 1024 at maximum
        if target_height > 1024:
            target_height = 1024
            target_width = int(target_height / person_aspect_ratio)

        # Resize images while preserving aspect ratio
        person_image = resize_image(HWC3(person_image), target_width, target_height)
        inpaint_mask = resize_image(HWC3(inpaint_mask), target_width, target_height)

        # Set the aspect ratio for the model
        aspect_ratio = f"{target_width}×{target_height}"

        # Prepare arguments for the image generation task
        args = [
            True,  # Input image checkbox
            Remove clothes, full naked, straight pose standing posing forward straight, perfect anatomy,  # Prompt for generating garment
            modules.config.default_prompt_negative,  # Negative prompt
            False,  # Advanced checkbox
            modules.config.default_styles,  # Style selections
            flags.Performance.QUALITY.value,  # Performance selection
            aspect_ratio,  # Aspect ratio selection
            1,  # Image number
            modules.config.default_output_format,  # Output format
            np.random.randint(constants.MIN_SEED, constants.MAX_SEED),  # Random seed
            modules.config.default_sample_sharpness,  # Sharpness
            modules.config.default_cfg_scale,  # Guidance scale
            modules.config.default_base_model_name,  # Base model
            modules.config.default_refiner_model_name,  # Refiner model
            modules.config.default_refiner_switch,  # Refiner switch
        ] + modules.config.default_loras + [
            True,  # Current tab (set to True for inpainting)
            "inpaint",  # Inpaint mode
            flags.disabled,  # UOV method
            None,  # UOV input image
            [],  # Outpaint selections
            {'image': person_image, 'mask': inpaint_mask},  # Inpaint input image
            prompt,  # Inpaint additional prompt
            inpaint_mask,  # Inpaint mask image
            True,  # Disable preview
            True,  # Disable intermediate results
            modules.config.default_black_out_nsfw,  # Black out NSFW
            1.5,  # Positive ADM guidance
            0.8,  # Negative ADM guidance
            0.3,  # ADM guidance end
            modules.config.default_cfg_tsnr,  # CFG mimicking from TSNR
            modules.config.default_sampler,  # Sampler name
            modules.config.default_scheduler,  # Scheduler name
            -1,  # Overwrite step
            -1,  # Overwrite switch
            target_width,  # Overwrite width
            target_height,  # Overwrite height
            -1,  # Overwrite vary strength
            modules.config.default_overwrite_upscale,  # Overwrite upscale strength
            False,  # Mixing image prompt and vary upscale
            True,  # Mixing image prompt and inpaint
            False,  # Debugging CN preprocessor
            False,  # Skipping CN preprocessor
            100,  # Canny low threshold
            200,  # Canny high threshold
            flags.refiner_swap_method,  # Refiner swap method
            0.5,  # ControlNet softness
            False,  # FreeU enabled
            1.0,  # FreeU b1
            1.0,  # FreeU b2
            1.0,  # FreeU s1
            1.0,  # FreeU s2
            False,  # Debugging inpaint preprocessor
            False,  # Inpaint disable initial latent
            modules.config.default_inpaint_engine_version,  # Inpaint engine
            1.0,  # Inpaint strength
            0.618,  # Inpaint respective field
            False,  # Inpaint mask upload checkbox
            False,  # Invert mask checkbox
            0,  # Inpaint erode or dilate
            modules.config.default_save_metadata_to_images,  # Save metadata to images
            modules.config.default_metadata_scheme,  # Metadata scheme
        ]

        # Create and append the image generation task
        task = worker.AsyncTask(args=args)
        worker.async_tasks.append(task)

        # Wait for the task to start and finish processing
        while not task.processing:
            time.sleep(0.1)
        while task.processing:
            time.sleep(0.1)

        # Check if results were generated successfully
        if task.results and isinstance(task.results, list) and len(task.results) > 0:
            result_path = task.results[0]
            if output_path:
                os.rename(result_path, output_path)
                result_path = output_path
            print(f"Generation successful. Result saved at: {result_path}")
            return result_path
        else:
            print("No results generated")
            return None

    except Exception as e:
        print(f"Error in virtual_try_on: {str(e)}")
        traceback.print_exc()
        return None

# Example usage
if __name__ == "__main__":
    input_image_path = "/Users/ikramali/projects/arbiosft_products/arbi-tryon/in_im/ate.png"
    output_image_path = "/Users/ikramali/projects/arbiosft_products/arbi-tryon/images/result.jpg"
    prompt = "wearing a red coat"
    category = "upper_body"  # or "lower_body" or "full_body"

    result = virtual_try_on(input_image_path, prompt, category, output_image_path)
    if result:
        print(f"Virtual try-on completed. Result saved at: {result}")
    else:
        print("Virtual try-on failed.")