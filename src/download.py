import os
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
MODEL_PATH = os.path.join(MODEL_DIR, "resnet50.pth")
LABELS_PATH = os.path.join(MODEL_DIR, "imagenet_classes.txt")

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download the model file
try:
    if not os.path.exists(MODEL_PATH):
        logging.info(f"Downloading model from {MODEL_URL}")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        logging.info(f"Model downloaded successfully to {MODEL_PATH}")
    else:
        logging.info("Model already exists. Skipping download.")
except Exception as e:
    logging.error(f"Error downloading model: {e}")

# Download the labels file
try:
    if not os.path.exists(LABELS_PATH):
        logging.info(f"Downloading labels from {LABELS_URL}")
        response = requests.get(LABELS_URL)
        with open(LABELS_PATH, "w") as f:
            f.write(response.text)
        logging.info(f"Labels downloaded successfully to {LABELS_PATH}")
    else:
        logging.info("Labels file already exists. Skipping download.")
except Exception as e:
    logging.error(f"Error downloading labels: {e}")
