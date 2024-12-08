import io
import logging
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import torch
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_model(model_path, labels_path):
    """
    Load a pre-trained model and its corresponding labels.
    """
    try:
        model = models.resnet50()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        with open(labels_path, 'r') as f:
            model_classes = [line.strip() for line in f]
        return model, model_classes
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def preprocess_image(image_bytes):
    """
    Preprocess the uploaded image for model inference.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise

def predict_image(model, model_classes, image_tensor, insight_image_path):
    """
    Perform both classification and facial analysis.
    """
    try:
        # Image Classification
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            probs, indices = torch.topk(probabilities, 3, dim=1)
            classification_results = [
                {"class": model_classes[idx.item()], "confidence": round(prob.item() * 100, 2)}
                for prob, idx in zip(probs[0], indices[0])
            ]

        # Facial Analysis
        app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
        img = ins_get_image(insight_image_path)
        faces = app.get(img)
        annotated_image_path = "output/faces_annotated.jpg"
        os.makedirs(os.path.dirname(annotated_image_path), exist_ok=True)
        cv2.imwrite(annotated_image_path, app.draw_on(img, faces))

        face_details = []
        for face in faces:
            face_data = {"bbox": face.bbox.tolist(), "landmark": face.landmark.tolist() if face.landmark else None}
            face_details.append(face_data)

        return {"classification": classification_results, "face_analysis": {"num_faces": len(faces), "details": face_details}}
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        raise