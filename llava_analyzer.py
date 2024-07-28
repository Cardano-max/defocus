# llava_analyzer.py

import torch
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.conversation import conv_templates, SeparatorStyle

class LLaVAImageAnalyzer:
    def __init__(self, model_path='liuhaotian/llava-v1.5-7b'):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.conv = None
        self.img_tensor = None
        self.roles = None
        self.stop_key = None
        self.load_models(model_path)

    def load_models(self, model_path):
        # Use MPS for Mac M1
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        quant_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type='nf8'
        )
        
        self.model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            device_map=device,
            quantization_config=quant_cfg
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        
        vision_tower = self.model.get_vision_tower()
        vision_tower.load_model()
        vision_tower.to(device=device)
        self.image_processor = vision_tower.image_processor
        disable_torch_init()

    def process_image(self, image):
        if isinstance(image, str):
            if image.startswith('http') or image.startswith('https'):
                response = requests.get(image)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            raise ValueError("Image must be a file path, URL, numpy array, or PIL Image object")
        
        self.img_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half()
        if torch.backends.mps.is_available():
            self.img_tensor = self.img_tensor.to('mps')
        return image

    def analyze_image(self, image, prompt):
        image = self.process_image(image)
        
        conv_mode = "v1"
        self.conv = conv_templates[conv_mode].copy()
        self.roles = self.conv.roles
        
        first_input = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt)
        self.conv.append_message(self.roles[0], first_input)
        self.conv.append_message(self.roles[1], None)
        
        if self.conv.sep_style == SeparatorStyle.TWO:
            self.stop_key = self.conv.sep2
        else:
            self.stop_key = self.conv.sep
        
        return self.generate_answer()

    def generate_answer(self):
        raw_prompt = self.conv.get_prompt()
        input_ids = tokenizer_image_token(raw_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        if torch.backends.mps.is_available():
            input_ids = input_ids.to('mps')
        
        stopping = KeywordsStoppingCriteria([self.stop_key], self.tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=self.img_tensor,
                stopping_criteria=[stopping],
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True
            )
        
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        self.conv.messages[-1][-1] = outputs
        return outputs.rsplit('</s>', 1)[0]

def analyze_person(image):
    analyzer = LLaVAImageAnalyzer()
    prompt = "Describe the person in this image in detail. Include their body type, estimated height, skin color, and any other notable features. Be very specific and technical in your description."
    return analyzer.analyze_image(image, prompt)

def analyze_garment(image):
    analyzer = LLaVAImageAnalyzer()
    prompt = "Describe the garment in this image in detail. Include its color (be specific, including shade), style, design, any logos or patterns, and the type of fabric if discernible. Be very specific and technical in your description."
    return analyzer.analyze_image(image, prompt)