# Face Attribute Detection with InsightFace and FastAPI

This project implements a Face Attribute Detection system that uses InsightFace for facial analysis and a ResNet50 model for image classification. The system is powered by FastAPI to provide a RESTful API for predictions, making it easy to integrate into other applications.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Model Details](#model-details)
- [App Architecture](#app-architecture)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Acknowledgments](#acknowledgments)

## Introduction

This repository is designed to detect facial attributes and classify images using two powerful machine learning tools:

- **InsightFace**: A high-performance library for face recognition and analysis.
- **ResNet50**: A deep residual network pre-trained on the ImageNet dataset for general image classification.

The system combines these tools to deliver a dual-purpose application capable of predicting image classes and detecting facial features.

## Features

- **Facial Analysis**: Detect faces, landmarks, and bounding boxes using InsightFace.
- **Image Classification**: Classify images into categories using ResNet50.
- **RESTful API**: Easy-to-use endpoints for predictions and health checks.
- **Configurable Paths**: Dynamic model and data paths via environment variables.

## Model Details

### ResNet50
- **Architecture**: A 50-layer residual network designed for image classification tasks.
- **Training Dataset**: Pre-trained on ImageNet, capable of predicting 1,000 categories.

### InsightFace
- **Purpose**: Provides robust facial detection and attribute analysis.
- **Capabilities**:
  - Face bounding boxes
  - Facial landmarks
  - Attribute detection (gender, age, etc., if supported by the model)

## App Architecture

The application is structured as follows:

1. **Download Module**:
   - Downloads the pre-trained ResNet50 model and ImageNet class labels.

2. **Inference Module**:
   - Loads the model and performs image classification.
   - Uses InsightFace for facial detection and analysis.

3. **API Module**:
   - Built using FastAPI to expose RESTful endpoints for inference.

### Workflow
1. User uploads an image.
2. The image is preprocessed and passed to the classification and facial analysis models.
3. Results are returned as a JSON response.

## Installation Instructions

### Prerequisites
Ensure the following are installed:
- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-attribute-detection.git
   cd face-attribute-detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Navigate to the `src` directory:
   ```bash
   cd src
   ```

5. Download models and labels:
   ```bash
   python download.py
   ```

6. Run the FastAPI server:
   ```bash
   python main.py
   ```

7. Access the API:
   - Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

## Usage

### Running Predictions
- **Health Check**: Verify the API is running:
  ```bash
  curl http://127.0.0.1:8000/health
  ```

- **Predict Attributes and Classify Image**:
  Use a tool like `curl` or Postman to upload an image:
  ```bash
  curl -X POST "http://127.0.0.1:8000/predict" \
       -H "Content-Type: multipart/form-data" \
       -F "file=@path_to_your_image.jpg"
  ```

- **Response**:
  ```json
  {
    "results": {
      "classification": [
        {"class": "class_name", "confidence": 95.5},
        {"class": "class_name", "confidence": 88.2},
        {"class": "class_name", "confidence": 78.1}
      ],
      "face_analysis": {
        "num_faces": 2,
        "details": [
          {"bbox": [x1, y1, x2, y2], "landmark": [[x1, y1], [x2, y2], ...]}
        ]
      }
    }
  }
  ```

## Project Structure
```
face-attribute-detection/
├── README.md             # Project documentation
├── requirements.txt      # Project dependencies
├── models/               # Directory for storing downloaded models
├── src/                  # Source code directory
│   ├── inference.py      # Model inference logic
│   ├── main.py           # FastAPI application
│   └── download.py       # Script to download models and labels
```

## API Endpoints

### `/health`
- **Method**: GET
- **Description**: Checks if the API is running.
- **Response**:
  ```json
  {"status": "API is running"}
  ```

### `/predict`
- **Method**: POST
- **Description**: Accepts an image and returns classification and face analysis results.
- **Request**: Multipart form data with image file.
- **Response**: JSON containing classification and facial analysis results.

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface): For providing robust facial analysis tools.
- [PyTorch](https://pytorch.org/): For pre-trained ResNet50 and other ML utilities.
- [FastAPI](https://fastapi.tiangolo.com/): For the easy-to-use web framework.

---
