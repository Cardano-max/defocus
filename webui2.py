import gradio as gr
import random
import time
import traceback
import sys
import os
import numpy as np
import modules.config
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
from modules.util import HWC3, resize_image, generate_temp_filename
from modules.private_logger import get_current_html_path, log
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io
import cv2
from queue import Queue
from threading import Lock, Event, Thread
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from Masking.masking import Masking
from modules.image_restoration import restore_image

# Garment processing and caching
from concurrent.futures import ThreadPoolExecutor
import hashlib

def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_cor = image_to_base64("images/corre.jpg")
base64_inc = image_to_base64("images/inc.jpg")

# Set up environment variables for sharing data
os.environ['GRADIO_PUBLIC_URL'] = ''
os.environ['GENERATED_IMAGE_PATH'] = ''
os.environ['MASKED_IMAGE_PATH'] = ''

def custom_exception_handler(exc_type, exc_value, exc_traceback):
    print("An unhandled exception occurred:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)

sys.excepthook = custom_exception_handler

# Initialize Masker
masker = Masking()

# Initialize queue and locks
task_queue = Queue()
queue_lock = Lock()
current_task_event = Event()
queue_update_event = Event()

# Garment cache
garment_cache = {}
garment_cache_lock = Lock()

# Function to process and cache garment image
def process_and_cache_garment(garment_image):
    garment_hash = hashlib.md5(garment_image.tobytes()).hexdigest()
    
    with garment_cache_lock:
        if garment_hash in garment_cache:
            return garment_cache[garment_hash]
    
    processed_garment = resize_image(HWC3(garment_image), 1024, 1024)
    
    with garment_cache_lock:
        garment_cache[garment_hash] = processed_garment
    
    return processed_garment

def send_feedback_email(rating, comment):
    sender_email = "feedback@arbitryon.com"
    receiver_email = "feedback@arbitryon.com"
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = f"ArbiTryOn Feedback - Rating: {rating}"

    body = f"Rating: {rating}/5\nComment: {comment}"
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("localhost", 1025) as server:
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("Feedback email sent successfully")
        return True
    except Exception as e:
        print(f"Failed to send feedback email: {str(e)}")
        return False

def check_image_quality(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    width, height = image.size
    resolution = width * height
    
    threshold = 512 * 512
    
    return resolution >= threshold

def virtual_try_on(clothes_image, person_image, category_input):
    try:
        processed_clothes = process_and_cache_garment(clothes_image)

        if not check_image_quality(person_image):
            print("Low resolution person image detected. Restoring...")
            person_image = restore_image(person_image)

        if not isinstance(person_image, Image.Image):
            person_pil = Image.fromarray(person_image)
        else:
            person_pil = person_image

        person_image_path = os.path.join(modules.config.path_outputs, f"person_image_{int(time.time())}.png")
        person_pil.save(person_image_path)
        print(f"User-uploaded person image saved at: {person_image_path}")

        categories = {
            "Upper Body": "upper_body",
            "Lower Body": "lower_body",
            "Full Body": "dresses"
        }
        print(f"Category Input: {category_input}")
        
        category = categories.get(category_input, "upper_body")
        print(f"Using category: {category}")
        
        inpaint_mask = masker.get_mask(person_pil, category=category)

        orig_person_h, orig_person_w = person_image.shape[:2]
        person_aspect_ratio = orig_person_h / orig_person_w
        target_width = 1024
        target_height = int(target_width * person_aspect_ratio)

        if target_height > 1024:
            target_height = 1024
            target_width = int(target_height / person_aspect_ratio)

        person_image = resize_image(HWC3(person_image), target_width, target_height)
        inpaint_mask = resize_image(HWC3(inpaint_mask), target_width, target_height)

        aspect_ratio = f"{target_width}×{target_height}"

        plt.figure(figsize=(10, 10))
        plt.imshow(inpaint_mask, cmap='gray')
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        masked_image_path = os.path.join(modules.config.path_outputs, f"masked_image_{int(time.time())}.png")
        with open(masked_image_path, 'wb') as f:
            f.write(buf.getvalue())
        
        plt.close()

        os.environ['MASKED_IMAGE_PATH'] = masked_image_path

        loras = []
        for lora in modules.config.default_loras:
            loras.extend(lora)

        args = [
            True,
            "",
            modules.config.default_prompt_negative,
            False,
            modules.config.default_styles,
            flags.Performance.QUALITY.value,
            aspect_ratio,
            1,
            modules.config.default_output_format,
            random.randint(constants.MIN_SEED, constants.MAX_SEED),
            modules.config.default_sample_sharpness,
            modules.config.default_cfg_scale,
            modules.config.default_base_model_name,
            modules.config.default_refiner_model_name,
            modules.config.default_refiner_switch,
        ] + loras + [
            True,
            "inpaint",
            flags.disabled,
            None,
            [],
            {'image': person_image, 'mask': inpaint_mask},
            "Wearing a new garment",
            inpaint_mask,
            True,
            True,
            modules.config.default_black_out_nsfw,
            1.5,
            0.8,
            0.3,
            modules.config.default_cfg_tsnr,
            modules.config.default_sampler,
            modules.config.default_scheduler,
            -1,
            -1,
            target_width,
            target_height,
            -1,
            modules.config.default_overwrite_upscale,
            False,
            True,
            False,
            False,
            100,
            200,
            flags.refiner_swap_method,
            0.5,
            False,
            1.0,
            1.0,
            1.0,
            1.0,
            False,
            False,
            modules.config.default_inpaint_engine_version,
            1.0,
            0.618,
            False,
            False,
            0,
            modules.config.default_save_metadata_to_images,
            modules.config.default_metadata_scheme,
        ]

        args.extend([
            processed_clothes,
            0.86,
            0.97,
            flags.default_ip,
        ])

        task = worker.AsyncTask(args=args)
        worker.async_tasks.append(task)

        while not task.processing:
            time.sleep(0.1)
        while task.processing:
            time.sleep(0.1)

        if task.results and isinstance(task.results, list) and len(task.results) > 0:
            return {"success": True, "image_path": task.results[0], "masked_image_path": masked_image_path, "person_image_path": person_image_path}
        else:
            return {"success": False, "error": "No results generated"}

    except Exception as e:
        print(f"Error in virtual_try_on: {str(e)}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

example_garments = [
    "images/b2.jpeg",
    "images/b4.jpeg",
    "images/b5.jpeg",
    "images/b6.jpeg",
    "images/b7.png",
    "images/b8.png",
    "images/b9.png",
    "images/b10.png",
    "images/b11.png",
    "images/b12.png",
    "images/b13.png",
    "images/b14.jpg",
    "images/b15.png",
    "images/b17.png",
    "images/b18.png",
    "images/t0.png",
    "images/1.png",
    "images/t2.png",
    "images/t3.png",
    "images/t4.png",
    "images/t5.png",
    "images/t6.png",
    "images/t7.png",
    "images/t16.png",
    "images/l19.png",
    "images/l20.png",
    "images/l4.png",
    "images/l5.png",
    "images/l7.png",
    "images/l8.png",
    "images/l10.jpeg",
    "images/l11.jpg",
    "images/l12.jpeg",
    "images/nine.jpeg",




]

with ThreadPoolExecutor(max_workers=4) as executor:
    example_garment_images = list(executor.map(lambda x: Image.open(x), example_garments))
    executor.map(process_and_cache_garment, example_garment_images)

loading_messages = [
    "Preparing your virtual fitting room...",
    "Analyzing fashion possibilities...",
    "Calculating style quotient...",
    "Assembling your digital wardrobe...",
    "Initiating virtual try-on sequence...",
]

error_messages = [
    "Oops! We've hit a small snag. Our team is on it!",
    "It seems our digital tailor needs a quick coffee break. We'll be right back!",
    "Minor hiccup in the fashion matrix. We're working to smooth it out.",
    "Looks like we're experiencing a slight style delay. Thanks for your patience!",
    "Our virtual dressing room is temporarily out of order. We're fixing it up!",
]

css = """
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    body, .gradio-container {
        font-family: 'Poppins', sans-serif;
        background-color: #f0f4f8;
        color: #333;
    }

    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }

    .header {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        padding: 2rem 0;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }

    .title {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1.2rem;
        font-weight: 300;
    }

    .main-content {
        display: flex;
        gap: 2rem;
        margin-bottom: 2rem;
    }

    .section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #4a4a4a;
    }

    .gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 1rem;
    }

    .gallery img {
        width: 100%;
        height: auto;
        border-radius: 5px;
        transition: transform 0.3s ease;
    }

    .gallery img:hover {
        transform: scale(1.05);
    }

    .upload-area {
        border: 2px dashed #a777e3;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .upload-area:hover {
        background-color: #f0e6ff;
    }

    .button-primary {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        padding: 0.8rem 1.5rem;
        border: none;
        border-radius: 5px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .button-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .result-area {
        display: flex;
        justify-content: space-between;
        margin-top: 2rem;
    }

    .result-image {
        width: 48%;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .feedback-area {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-top: 2rem;
    }

    .loading {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #a777e3;
        animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }
"""

def process_queue():
    while True:
        task = task_queue.get()
        if task is None:
            break
        clothes_image, person_image, category_input, result_callback = task
        current_task_event.set()
        queue_update_event.set()
        result = virtual_try_on(clothes_image, person_image, category_input)
        current_task_event.clear()
        result_callback(result)
        task_queue.task_done()
        queue_update_event.set()

# Start the queue processing thread
queue_thread = Thread(target=process_queue, daemon=True)
queue_thread.start()

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <div class="header">
            <div class="container">
                <h1 class="title">ArbiTryOn: Virtual Fitting Room</h1>
                <p class="subtitle">Experience the future of online shopping with our AI-powered try-on system</p>
            </div>
        </div>
        """
    )

    with gr.Row(class_name="container main-content"):
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### Step 1: Select a Garment", class_name="section-title")
            example_garment_gallery = gr.Gallery(value=example_garments, columns=3, rows=3, label="Example Garments", elem_class="gallery")
            clothes_input = gr.Image(label="Selected Garment", source="upload", type="numpy", elem_classes="upload-area")

        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### Step 2: Upload Your Photo", class_name="section-title")
            person_input = gr.Image(label="Your Photo", source="upload", type="numpy", elem_classes="upload-area")
        
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### Step 3: Select Category", class_name="section-title")
            category_input = gr.Radio(
                choices=["Upper Body", "Lower Body", "Full Body"],
                label="Select a Category",
                value="Upper Body"
            )

    gr.HTML(f"""
        <div class="container">
            <div class="section">
                <h3 class="section-title">Posing Instructions</h3>
                <div style="display: flex; justify-content: space-between;">
                    <div style="text-align: center; width: 48%;">
                        <img src="data:image/jpg;base64,{base64_cor}" alt="Correct pose" style="width: 100%; border-radius: 10px;">
                        <p>✅ Correct: Neutral pose, facing forward</p>
                    </div>
                    <div style="text-align: center; width: 48%;">
                        <img src="data:image/jpeg;base64,{base64_inc}" alt="Incorrect pose" style="width: 100%; border-radius: 10px;">
                        <p>❌ Incorrect: Angled or complex pose</p>
                    </div>
                </div>
            </div>
        </div>
    """)

    with gr.Row(class_name="container"):
        try_on_button = gr.Button("Try It On!", elem_classes="button-primary")

    with gr.Row(class_name="container"):
        loading_indicator = gr.HTML(visible=False)
        status_info = gr.HTML(visible=False, elem_classes="section")
        error_output = gr.HTML(visible=False, elem_classes="section")

    with gr.Row(class_name="container result-area", visible=False) as result_row:
        masked_output = gr.Image(label="Mask Visualization", elem_classes="result-image")
        try_on_output = gr.Image(label="Virtual Try-On Result", elem_classes="result-image")

    image_link = gr.HTML(visible=True, elem_classes="container")

    with gr.Row(class_name="container feedback-area", visible=False) as feedback_row:
        rating = gr.Slider(minimum=1, maximum=5, step=1, label="Rate your experience (1-5 stars)")
        comment = gr.Textbox(label="Leave a comment (optional)")
        submit_feedback = gr.Button("Submit Feedback", elem_classes="button-primary")

    feedback_status = gr.HTML(visible=False, elem_classes="container")

    def select_example_garment(evt: gr.SelectData):
        return example_garments[evt.index]

    example_garment_gallery.select(select_example_garment, None, clothes_input)

    def process_virtual_try_on(clothes_image, person_image, category_input):
        if clothes_image is None or person_image is None:
            yield {
                loading_indicator: gr.update(visible=False),
                status_info: gr.update(visible=False),
                result_row: gr.update(visible=False),
                error_output: gr.update(value="<p>Please upload both a garment image and a person image to proceed.</p>", visible=True),
                image_link: gr.update(visible=False),
                feedback_row: gr.update(visible=False)
            }
            return

        if current_task_event.is_set():
            yield {
                loading_indicator: gr.update(visible=True, value=f"<div class='loading'></div><p>{random.choice(loading_messages)}</p>"),
                status_info: gr.update(value="<p>Your request has been queued. We'll process it as soon as possible.</p>", visible=True),
                result_row: gr.update(visible=False),
                error_output: gr.update(visible=False),
                image_link: gr.update(visible=False),
            }

        def result_callback(result):
            nonlocal generation_done, generation_result
            generation_done = True
            generation_result = result

        with queue_lock:
            current_position = task_queue.qsize()
            task_queue.put((clothes_image, person_image, category_input, result_callback))

        generation_done = False
        generation_result = None

        while not generation_done:
            if current_task_event.is_set() and current_position == 0:
                yield {
                    loading_indicator: gr.update(visible=True, value=f"<div class='loading'></div><p>{random.choice(loading_messages)}</p>"),
                    status_info: gr.update(value="<p>Processing your request. This may take a few minutes.</p>", visible=True),
                    result_row: gr.update(visible=False),
                    error_output: gr.update(visible=False),
                    image_link: gr.update(visible=False),
                }
            elif current_position > 0:
                yield {
                    loading_indicator: gr.update(visible=True, value=f"<div class='loading'></div><p>{random.choice(loading_messages)}</p>"),
                    status_info: gr.update(value=f"<p>Your request is in queue. Current position: {current_position}</p><p>Estimated wait time: {current_position * 2} minutes</p>", visible=True),
                    result_row: gr.update(visible=False),
                    error_output: gr.update(visible=False),
                    image_link: gr.update(visible=False),
                }
            else:
                yield {
                    loading_indicator: gr.update(visible=True, value=f"<div class='loading'></div><p>{random.choice(loading_messages)}</p>"),
                    status_info: gr.update(value="<p>Your request is next in line. Processing will begin shortly.</p>", visible=True),
                    result_row: gr.update(visible=False),
                    error_output: gr.update(visible=False),
                    image_link: gr.update(visible=False),
                }
            
            queue_update_event.wait(timeout=5)
            queue_update_event.clear()
            current_position = max(0, task_queue.qsize() - 1)

        if generation_result is None:
            yield {
                loading_indicator: gr.update(visible=False),
                status_info: gr.update(visible=False),
                result_row: gr.update(visible=False),
                error_output: gr.update(value=f"<p>{random.choice(error_messages)}</p><p>Remember, we're still in beta. We appreciate your understanding as we work to improve our service.</p>", visible=True),
                image_link: gr.update(visible=False),
                feedback_row: gr.update(visible=False)
            }
        elif generation_result['success']:
            generated_image_path = generation_result['image_path']
            masked_image_path = generation_result['masked_image_path']
            person_image_path = generation_result['person_image_path']
            gradio_url = os.environ.get('GRADIO_PUBLIC_URL', '')

            if gradio_url and generated_image_path and masked_image_path and person_image_path:
                output_image_link = f"{gradio_url}/file={generated_image_path}"
                masked_image_link = f"{gradio_url}/file={masked_image_path}"
                person_image_link = f"{gradio_url}/file={person_image_path}"
                link_html = f'<div class="section"><h3 class="section-title">Result Links</h3><p><a href="{output_image_link}" target="_blank">View Try-On Result</a> | <a href="{masked_image_link}" target="_blank">View Mask Visualization</a> | <a href="{person_image_link}" target="_blank">View Original Person Image</a></p></div>'

                yield {
                    loading_indicator: gr.update(visible=False),
                    status_info: gr.update(value="<p>Your virtual try-on is complete! Check out the results below.</p>", visible=True),
                    result_row: gr.update(visible=True),
                    masked_output: gr.update(value=masked_image_path),
                    try_on_output: gr.update(value=generated_image_path),
                    image_link: gr.update(value=link_html, visible=True),
                    error_output: gr.update(visible=False),
                    feedback_row: gr.update(visible=True)
                }
            else:
                yield {
                    loading_indicator: gr.update(visible=False),
                    status_info: gr.update(visible=False),
                    result_row: gr.update(visible=False),
                    error_output: gr.update(value="<p>We encountered an issue while generating your try-on results. Our team has been notified and is working on a solution. Please try again later.</p>", visible=True),
                    image_link: gr.update(visible=False),
                    feedback_row: gr.update(visible=False)
                }
        else:
            yield {
                loading_indicator: gr.update(visible=False),
                status_info: gr.update(visible=False),
                result_row: gr.update(visible=False),
                error_output: gr.update(value=f"<p>An error occurred: {generation_result['error']}</p><p>Our team has been notified and is working on a solution. We appreciate your patience as we improve our beta service.</p>", visible=True),
                image_link: gr.update(visible=False),
                feedback_row: gr.update(visible=False)
            }

    try_on_button.click(
        process_virtual_try_on,
        inputs=[clothes_input, person_input, category_input],
        outputs=[loading_indicator, status_info, result_row, masked_output, try_on_output, image_link, error_output, feedback_row]
    )

    def submit_user_feedback(rating, comment):
        if send_feedback_email(rating, comment):
            return gr.update(value="Thank you for your feedback!", visible=True)
        else:
            return gr.update(value="Failed to submit feedback. Please try again later.", visible=True)

    submit_feedback.click(
        submit_user_feedback,
        inputs=[rating, comment],
        outputs=feedback_status
    )

demo.queue()

def custom_launch():
    app, local_url, share_url = demo.launch(share=True, prevent_thread_lock=True)
    
    if share_url:
        os.environ['GRADIO_PUBLIC_URL'] = share_url
        print(f"Running on public URL: {share_url}")
    
    return app, local_url, share_url

custom_launch()

# Keep the script running
while True:
    time.sleep(1)