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
    prompt = (
        "Describe the person in this image in detail. Include the following information: "
        "1. Gender: Specify the person's gender. "
        "2. Age: Estimate the person's age range. "
        "3. Body proportions: Describe the body proportions (e.g., long legs, broad shoulders). "
        "4. Body type: Describe the body type (e.g., athletic, slender, muscular). "
        "5. Body weight: Estimate the body weight or describe if the person appears to be underweight, normal weight, or overweight. "
        "6. Height: Provide an estimated height in feet or centimeters. "
        "7. Skin color: Describe the skin tone and any noticeable features such as freckles or scars. "
        "8. Clothing: Detail the type of clothing, its color, and any notable design elements. "
        "9. Posture and stance: Describe the person's posture and stance (e.g., standing straight, slouched). "
        "10. Notable features: Highlight any other notable features or distinguishing characteristics. "
        "Be very specific and technical in your description."
    )
    return analyze_image(image, prompt)

def analyze_garment(image):
    prompt = (
        "Describe the garment in this image in detail. Include the following information: "
        "1. Type: Specify the exact type of garment (e.g., blazer, dress shirt, trousers). "
        "2. Material: Identify the fabric type and texture. "
        "3. Color: Describe the color in detail, including the shade. "
        "4. Size: Estimate the size (e.g., S, M, L, XL) and provide measurements if possible. "
        "5. Fit: Describe the fit of the garment (e.g., slim fit, loose). "
        "6. Design elements: Note any design elements such as buttons, zippers, or embroidery. "
        "7. Pattern: Describe any patterns or prints in detail (e.g., stripes, polka dots). "
        "8. Brand/Logo: Identify any visible branding or logos. "
        "9. Multi-piece outfits: For multi-piece outfits, describe each piece separately (top, bottom, etc.). "
        "10. Additional details: Include any other relevant information about the garment's style, design, or fabric. "
        "Be very specific and technical in your description."
    )
    return analyze_image(image, prompt)
