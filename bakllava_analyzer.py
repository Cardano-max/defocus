# bakllava_analyzer.py

import requests
import base64
from PIL import Image
import io
import json

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image, prompt):
    if isinstance(image, str):
        # If image is a file path
        base64_image = encode_image(image)
    elif isinstance(image, Image.Image):
        # If image is a PIL Image object
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        raise ValueError("Image must be a file path or PIL Image object")

    payload = {
        "model": "bakllava",
        "prompt": prompt,
        "stream": False,
        "images": [base64_image]
    }

    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        # Parse the response carefully
        response_text = response.text.strip()
        try:
            response_json = json.loads(response_text)
            return response_json.get('response', '')
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw text
            return response_text
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def analyze_person(image):
    prompt = "Describe the person in this image in detail. Include their Gender, Age, Body proportions(Use a technical ratio (e.g., 6:9 for height to width)),  body type, Body Weight(Estimate in pounds and provide a BMI category),Height(Apprx:Estimate in feet and inches),Skin color:Use a specific color code (e.g., Fitzpatrick scale),Clothing,Posture and stance, estimated height, skin color, and any other notable features. Be very specific and technical in your description."
    return analyze_image(image, prompt)

def analyze_garment(image):
    prompt = "Describe the garment in this image in detail. Include its Type:Specify the exact type of garment (e.g., blazer, dress shirt, trousers), Material: Identify the fabric type and texture, color (be specific, including shade),Size: Estimate the size (e.g., S, M, L, XL) and provide measurements,Fit,Design elements, Pattern: Describe any patterns or prints in detail,,Brand/Logo: Identify any visible branding or logos,For multi-piece outfits: Describe each piece separately (top, bottom, etc.),, style, design, any logos or patterns, and the type of fabric if discernible. Be very specific and technical in your description."
    return analyze_image(image, prompt)