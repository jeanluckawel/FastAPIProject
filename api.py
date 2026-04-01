import numpy as np
from fastapi import FastAPI, UploadFile
from tensorflow.keras.models import load_model
import io
from PIL import Image

app = FastAPI()


model = load_model("face_classification_cnn_model.h5", compile=False)

def preprocess(img):
    img = img.resize((64, 64))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(file: UploadFile):
    img_data = await file.read()
    img = Image.open(io.BytesIO(img_data))
    img = preprocess(img)

    pred = model.predict(img)

    classes = ["kafanda", "kawel", "kalonda"]
    pred_index = int(np.argmax(pred))
    pred_class = classes[pred_index]

    return {
        "prediction": pred_class,
        "confidence": float(pred[0][pred_index])
    }