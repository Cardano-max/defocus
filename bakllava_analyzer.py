# bakllava_analyzer.py

import requests
import base64
from PIL import Image
import io
import json
import numpy as np

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image, prompt):
    if isinstance(image, str):
        base64_image = encode_image(image)
    elif isinstance(image, Image.Image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        raise ValueError("Image must be a file path, PIL Image object, or numpy array")

    payload = {
        "model": "bakllava",
        "prompt": prompt,
        "stream": False,
        "images": [base64_image]
    }

    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        response_text = response.text.strip()
        try:
            response_json = json.loads(response_text)
            return response_json.get('response', '')
        except json.JSONDecodeError:
            return response_text
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def analyze_person(image):
    prompt = """
    Analyze the person in this image with extreme precision. Provide a technical description including:
    1. Gender: Specify male or female.
    2. Age: Estimate the age range.
    3. Body Proportions: Use a technical ratio (e.g., 6:9 for waist-to-height).
    4. Weight: Estimate in pounds and provide a BMI category.
    5. Height: Estimate in feet and inches.
    6. Skin Color: Use a precise color description (e.g., RGB values or Pantone code).
    7. Fitness Level: Describe muscle tone and body fat percentage.
    8. Posture: Analyze standing position and alignment.
    9. Facial Features: Describe eye color, hair color and style, facial hair if any.
    10. Clothing: Describe any visible clothing not part of the target garment.
    
    Format the response as a structured list with these headings.
    """
    return analyze_image(image, prompt)

def analyze_garment(image):
    prompt = """
    Provide an extremely detailed technical analysis of the garment in this image:
    1. Type: Specify the exact type (e.g., blazer, dress shirt, etc.).
    2. Color: Use precise color codes (RGB and Pantone).
    3. Material: Identify the fabric type and composition.
    4. Size: Estimate the size (S, M, L, XL, XXL, etc.) based on proportions.
    5. Pattern/Design: Describe any patterns, prints, or unique design elements.
    6. Fit: Analyze how it would fit on a body (slim, regular, loose, etc.).
    7. Details: Note buttons, zippers, pockets, cuffs, collars, etc.
    8. Brand/Logo: Identify any visible branding or logos.
    9. Condition: Assess the condition of the garment.
    10. Style: Categorize the style (formal, casual, business, etc.).
    11. For multi-piece outfits:
        a. Top: Describe in detail.
        b. Bottom: Describe pants, skirt, etc. if visible.
        c. Accessories: Note any visible accessories.
    
    Format the response as a structured list with these headings.
    """
    return analyze_image(image, prompt)