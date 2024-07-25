# Use the official Python 3.10 image as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y curl libgl1 libglib2.0-0 build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy all the files from the current directory to the container's working directory
COPY . /app

# Install any necessary dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose any ports the app is expected to run on
EXPOSE 8000

# Define the command to run the application
CMD ["python", "entry_with_update.py", "--disable-offload-from-vram", "--always-high-vram", "--theme", "dark", "--share", "--all-in-fp32", "--vae-in-fp32", "--clip-in-fp32", "--attention-split", "--debug-mode"]