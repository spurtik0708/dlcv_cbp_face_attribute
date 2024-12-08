import os
import logging
from fastapi import FastAPI, File, UploadFile
from inference import load_model, preprocess_image, predict_image

# FastAPI Application
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "resnet50.pth")
LABELS_PATH = os.path.join(MODEL_DIR, "imagenet_classes.txt")

# Load the model and labels
try:
    model, model_classes = load_model(MODEL_PATH, LABELS_PATH)
    logging.info("Model and labels loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model and labels: {e}")
    raise

@app.get("/health")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)
        results = predict_image(model, model_classes, image_tensor, insight_image_path="t1")
        return {"results": results}
    except Exception as e:
        logging.error(f"Error processing prediction request: {e}")
        return {"error": "Prediction failed."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

