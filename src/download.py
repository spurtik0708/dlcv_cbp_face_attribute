import os
import requests

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
MODEL_PATH = os.path.join(MODEL_DIR, "resnet50.pth")
LABELS_PATH = os.path.join(MODEL_DIR, "imagenet_classes.txt")

os.makedirs(MODEL_DIR, exist_ok=True)

# Download Model
try:
    if not os.path.exists(MODEL_PATH):
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        logging.info("Model downloaded successfully.")
    else:
        logging.info("Model already exists.")
except Exception as e:
    logging.error(f"Error downloading model: {e}")

# Download Labels
try:
    if not os.path.exists(LABELS_PATH):
        response = requests.get(LABELS_URL)
        with open(LABELS_PATH, "w") as f:
            f.write(response.text)
        logging.info("Labels downloaded successfully.")
    else:
        logging.info("Labels file already exists.")
except Exception as e:
    logging.error(f"Error downloading labels: {e}")
