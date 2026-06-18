from contextlib import asynccontextmanager
from typing import Optional
from pydantic import Field, field_validator
import logging
import os
from typing import List
from dotenv import load_dotenv
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException, File
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import io
from PIL import Image
from ultralytics import YOLO
from medical_predictor import MedicalDiagnosisPredictor

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------

class PatientInput(BaseModel):
    age:              int   = Field(alias="Âge",              ge=0,   le=120)
    temperature:      float = Field(alias="Température",      ge=34.0, le=43.0)
    spo2:             float = Field(alias="SpO2",             ge=50.0, le=100.0)
    poids:            float = Field(alias="Poids",            ge=1.0,  le=300.0)
    pouls:            float = Field(alias="Pouls",            ge=20.0, le=300.0)
    systolic_bp:      float = Field(alias="Systolic_BP",      ge=50.0, le=300.0)
    diastolic_bp:     float = Field(alias="Diastolic_BP",     ge=20.0, le=200.0)
    sexe:             str   = Field(alias="Sexe")
    etat_civil:       str   = Field(alias="État Civil")
    etat_general:     str   = Field(alias="État Général")
    capacite_physique:str   = Field(alias="Capacité Physique")
    combined_text:    str   = Field(alias="combined_text",    min_length=3)

    model_config = {"populate_by_name": True}

    @field_validator("sexe")
    @classmethod
    def validate_sexe(cls, v: str) -> str:
        if v.upper() not in ("M", "F"):
            raise ValueError("Sexe doit être 'M' ou 'F'")
        return v.upper()

    @field_validator("etat_general")
    @classmethod
    def validate_etat_general(cls, v: str) -> str:
        allowed = {"Conservé", "Altéré"}
        if v not in allowed:
            raise ValueError(f"État Général doit être l'un de: {allowed}")
        return v

    @field_validator("capacite_physique")
    @classmethod
    def validate_capacite_physique(cls, v: str) -> str:
        allowed = {"Top", "Moyen", "Bas"}
        if v not in allowed:
            raise ValueError(f"Capacité Physique doit être l'un de: {allowed}")
        return v

    @field_validator("systolic_bp", "diastolic_bp")
    @classmethod
    def validate_bp_coherence(cls, v: float, info) -> float:
        # cross-field check done below in model_validator
        return v

    def to_predictor_dict(self) -> dict:
        """Return the dict with French aliases expected by MedicalDiagnosisPredictor."""
        return self.model_dump(by_alias=True)


class ProbabilityOutput(BaseModel):
    other_diagnosis:    float
    specific_diagnosis: float


class AdviseResponse(BaseModel):
    prediction:          int
    diagnosis_type:      str
    probabilities:       ProbabilityOutput
    ml_confidence:       float
    final_confidence:    Optional[float]
    clinical_flags:      list[str]
    clinical_override:   bool
    recommendation:      str
    override_reason:     Optional[str] = None


# ------------------------------------------------------------------
# Lifespan — load predictor once at startup
# ------------------------------------------------------------------

medical_predictor: MedicalDiagnosisPredictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global medical_predictor
    try:
        medical_predictor = MedicalDiagnosisPredictor()
        logger.info("MedicalDiagnosisPredictor loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load MedicalDiagnosisPredictor: {e}")
        raise
    yield


app = FastAPI(lifespan=lifespan)

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

def yolow_preprocess(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = np.array(img)

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
detection_model = YOLO(DETECTION_MODEL_PATH)

@app.post("/detect")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    img_array = yolow_preprocess(image)

    # Run inference
    results = detection_model(img_array)

    # Format detections to JSON
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": detection_model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0]]
            })

    return {"detections": detections}


@app.post("/advise", response_model=AdviseResponse)
async def advise(data: PatientInput):
    """
    Medical diagnosis advisory endpoint.
    Combines ML prediction with a rule-based clinical safety layer.
    Returns class 1 override if multiple critical vital signs are detected.
    """
    if medical_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle médical non disponible. Veuillez réessayer plus tard."
        )

    # cross-field BP coherence check
    if data.diastolic_bp >= data.systolic_bp:
        raise HTTPException(
            status_code=422,
            detail="Diastolic_BP doit être strictement inférieure à Systolic_BP."
        )

    try:
        result = medical_predictor.predict(data.to_predictor_dict())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"/advise prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors de la prédiction. Veuillez contacter l'administrateur."
        )

    return result