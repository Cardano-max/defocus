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
    Analyze the person in this image with extreme precision. Provide a detailed technical description including:
    1. Gender: Specify male or female.
    2. Age: Estimate the age range.
    3. Body proportions: Use a technical ratio (e.g., 6:9 for height to width).
    4. Weight: Estimate in pounds and provide a BMI category.
    5. Height: Estimate in feet and inches.
    6. Skin color: Use a specific color code (e.g., Fitzpatrick scale).
    7. Fitness level: Describe using specific terms (e.g., athletic, overweight, muscular).
    8. Facial features: Describe prominent features.
    9. Hair: Color, length, and style.
    10. Clothing: Describe what the person is wearing in detail.
    11. Posture and stance: Describe how the person is positioned.
    12. Background: Briefly mention the setting.
    Provide this information in a structured, bullet-point format.
    """
    return analyze_image(image, prompt)

def analyze_garment(image):
    prompt = """
    Analyze the garment in this image with extreme precision. Provide a detailed technical description including:
    1. Type: Specify the exact type of garment (e.g., blazer, dress shirt, trousers).
    2. Color: Use specific color codes (e.g., Pantone or RGB values) for all visible colors.
    3. Material: Identify the fabric type and texture.
    4. Size: Estimate the size (e.g., S, M, L, XL) and provide measurements if possible.
    5. Fit: Describe how it fits (e.g., slim, regular, loose).
    6. Design elements: Describe collars, cuffs, buttons, pockets, etc.
    7. Pattern: Describe any patterns or prints in detail.
    8. Brand/Logo: Identify any visible branding or logos.
    9. Condition: Describe the condition of the garment.
    10. Style: Categorize the style (e.g., formal, casual, streetwear).
    11. Accessories: Describe any accompanying accessories.
    12. For multi-piece outfits: Describe each piece separately (top, bottom, etc.).
    Provide this information in a structured, bullet-point format.
    """
    return analyze_image(image, prompt)