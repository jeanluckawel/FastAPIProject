import os
from typing import List

from dotenv import load_dotenv
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException, File
from starlette.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import io
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

origins = [
    "https://gjp-face-reconizer.test",  # Herd domain
    "http://localhost:3000",
    "http://localhost:8000",            # Default Laravel's Domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

MODEL_PATH = os.getenv("CURRENT_MODEL")

model = load_model(MODEL_PATH, compile=False)
IMG_SIZE = model.input_shape[1:3]
CLASSES = ["kafanda", "kawel", "kalonda"]

def preprocess(img):
    img = img.convert("RGB")

    img = img.resize(IMG_SIZE)
    img = np.array(img)
    img = np.array(img) / 255.0

    img = np.expand_dims(img, axis=0)
    return img

@app.get("/")
async def root():
    return {"message": "Future Start Now"}

@app.post("/predict")
async def predict(file: UploadFile = None):
    try:
        # ✅ 1. Check if file is provided
        print(file)
        if file is None or file.filename == "":
            raise HTTPException(status_code=400, detail="No file uploaded")

        # ✅ 2. Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # ✅ 3. Read & preprocess
        img_data = await file.read()
        img = Image.open(io.BytesIO(img_data))

        # ✅ 3.1 Preprocess
        img = preprocess(img)


        # ✅ 4. Prediction
        pred = model.predict(img, verbose=False)[0]

        # ✅ 5. Top prediction
        pred_index = int(np.argmax(pred))
        pred_class = CLASSES[pred_index]
        confidence = float(pred[pred_index])

        # ✅ 6. Top-3 predictions (better UX)
        top_indices = np.argsort(pred)[::-1][:3]
        top_predictions: List[dict] = [
            {
                "class": CLASSES[i],
                "confidence": float(pred[i])
            }
            for i in top_indices
        ]

        return {
            "prediction": pred_class,
            "confidence": confidence,
            "top_predictions": top_predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


DETECTION_MODEL_PATH = os.getenv("DETECTION_MODEL")

# Load your custom trained model
model = YOLO(DETECTION_MODEL_PATH)

@app.post("/detect")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    img_array = np.array(image)

    # Run inference
    results = model(img_array)

    # Format detections to JSON
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0]]
            })

    return {"detections": detections}