import time
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image

# Load the model and processor
processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")

# Load the image
image = Image.open("images/b3.jpeg")

# Measure the time taken to process the image and get the model outputs
start_time = time.time()

# Process the image and get inputs for the model
inputs = processor(images=image, return_tensors="pt")

# Get the model outputs
outputs = model(**inputs)

end_time = time.time()

# Calculate the total time taken
time_taken = end_time - start_time
print(f"Time taken to process the image and get the model outputs: {time_taken} seconds")
