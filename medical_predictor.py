import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalDiagnosisPredictor:
    """
    Medical Diagnosis Prediction Model for Deployment

    Combines ML prediction with a rule-based clinical safety layer.
    The clinical layer acts as a hard override for critical vital signs,
    independent of the ML model output.
    """

    CLINICAL_THRESHOLDS = {
        'SpO2_critical':       ('SpO2',        'lt', 94,  "SpO2 critique (<94%)"),
        'tachycardia':         ('Pouls',        'gt', 110, "Tachycardie sévère (>110 bpm)"),
        'hypotension':         ('Systolic_BP',  'lt', 90,  "Hypotension (systolique <90 mmHg)"),
        'hypertension':        ('Systolic_BP',  'gt', 180, "Hypertension sévère (systolique >180 mmHg)"),
        'hyperthermia':        ('Température',  'gt', 39.5,"Hyperthermie (>39.5°C)"),
        'hypothermia':         ('Température',  'lt', 35.5,"Hypothermie (<35.5°C)"),
        'altered_general':     ('État Général', 'eq', 'Altéré', "État général altéré"),
        'low_capacity':        ('Capacité Physique', 'eq', 'Bas', "Capacité physique basse"),
    }

    CLINICAL_OVERRIDE_THRESHOLD = 3  # nb of flags triggering override

    def __init__(self, model_path: str = "models/llm/best_medical_diagnosis_model.pkl"):
        """Initialize the predictor with the trained model."""
        resolved_path = os.getenv("CMEDICAL_LLM_MODEL", model_path)
        logger.info(f"Loading model from: {resolved_path}")
        self.model = joblib.load(resolved_path)
        self.required_features = [
            'Âge', 'Température', 'SpO2', 'Poids', 'Pouls',
            'Systolic_BP', 'Diastolic_BP', 'Sexe', 'État Civil',
            'État Général', 'Capacité Physique', 'combined_text'
        ]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_input(self, patient_data: Dict) -> bool:
        """Validate that all required features are present."""
        missing = [f for f in self.required_features if f not in patient_data]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return True

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_input(self, patient_data: Dict) -> pd.DataFrame:
        """Convert input dictionary to DataFrame format expected by model."""
        df = pd.DataFrame([patient_data])
        numerical_cols = [
            'Âge', 'Température', 'SpO2', 'Poids', 'Pouls',
            'Systolic_BP', 'Diastolic_BP'
        ]
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    # ------------------------------------------------------------------
    # Clinical safety layer
    # ------------------------------------------------------------------

    def _evaluate_flags(self, patient_data: Dict) -> List[str]:
        """
        Evaluate rule-based clinical flags against CLINICAL_THRESHOLDS.
        Returns list of triggered flag messages.
        """
        flags = []
        ops = {
            'lt': lambda a, b: a < b,
            'gt': lambda a, b: a > b,
            'eq': lambda a, b: a == b,
        }
        for _, (field, op, threshold, message) in self.CLINICAL_THRESHOLDS.items():
            value = patient_data.get(field)
            if value is not None:
                try:
                    if ops[op](value, threshold):
                        flags.append(message)
                except TypeError:
                    pass  # type mismatch — skip gracefully
        return flags

    def _clinical_override(self, flags: List[str], prediction: int) -> Tuple[bool, str]:
        """
        Returns (override_triggered, override_reason).
        Override fires only when prediction is 0 but flags reach the threshold.
        """
        if prediction == 0 and len(flags) >= self.CLINICAL_OVERRIDE_THRESHOLD:
            reason = (
                f"Override clinique activé ({len(flags)} signaux critiques détectés) — "
                "la prédiction ML a été écartée au profit de la sécurité patient."
            )
            return True, reason
        return False, ""

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def _get_recommendation(
        self,
        prediction: int,
        confidence: float,
        n_flags: int,
        override: bool
    ) -> str:
        if override or (prediction == 0 and n_flags >= self.CLINICAL_OVERRIDE_THRESHOLD):
            return (
                "URGENCE: constantes vitales critiques multiples — "
                "évaluation médicale immédiate requise indépendamment de la prédiction ML."
            )
        if prediction == 1:
            if confidence > 0.8:
                return "Priorité haute — attention médicale immédiate et diagnostic spécifique requis."
            return "Priorité moyenne — évaluation médicale complémentaire recommandée."
        # prediction == 0
        if confidence > 0.9:
            return "Priorité basse — consultation de routine ou soins habituels."
        return (
            "Incertain — confiance insuffisante pour exclure une pathologie spécifique. "
            "Évaluation médicale recommandée."
        )

    # ------------------------------------------------------------------
    # Public predict
    # ------------------------------------------------------------------

    def predict(self, patient_data: Dict) -> Dict:
        """
        Run full prediction pipeline:
          1. Validate input
          2. Run ML model
          3. Evaluate clinical flags
          4. Apply safety override if needed
          5. Return structured result
        """
        self.validate_input(patient_data)
        df = self.preprocess_input(patient_data)

        ml_prediction   = int(self.model.predict(df)[0])
        probabilities   = self.model.predict_proba(df)[0]
        ml_confidence   = float(max(probabilities))

        flags                    = self._evaluate_flags(patient_data)
        override, override_reason = self._clinical_override(flags, ml_prediction)

        final_prediction = 1 if override else ml_prediction
        final_confidence = ml_confidence if not override else None

        recommendation = self._get_recommendation(
            final_prediction, ml_confidence, len(flags), override
        )

        result = {
            "prediction": final_prediction,
            "diagnosis_type": (
                "Specific Diagnosis" if final_prediction == 1
                else "General/Other Diagnosis"
            ),
            "probabilities": {
                "other_diagnosis":    float(probabilities[0]),
                "specific_diagnosis": float(probabilities[1]),
            },
            "ml_confidence":    ml_confidence,
            "final_confidence": final_confidence,
            "clinical_flags":   flags,
            "clinical_override": override,
            "recommendation":   recommendation,
        }

        if override:
            result["override_reason"] = override_reason

        logger.warning(
            f"Clinical override triggered — flags: {flags}"
        ) if override else logger.info(
            f"Prediction={final_prediction}, confidence={ml_confidence:.3f}, flags={len(flags)}"
        )

        return result


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------

if __name__ == "__main__":
    predictor = MedicalDiagnosisPredictor()

    # Case that previously produced a false negative
    shock_patient = {
        'Âge': 35,
        'Température': 40.1,
        'SpO2': 90.0,
        'Poids': 60.0,
        'Pouls': 125.0,
        'Systolic_BP': 85.0,
        'Diastolic_BP': 55.0,
        'Sexe': 'F',
        'État Civil': 'Célibataire',
        'État Général': 'Altéré',
        'Capacité Physique': 'Bas',
        'combined_text': (
            "Fièvre très élevée avec frissons intenses, douleurs abdominales diffuses "
            "et ictère depuis 3 jours. Patiente en provenance d'une zone palustre, "
            "état de choc, oligurie."
        )
    }

    result = predictor.predict(shock_patient)

    print("\n=== Prediction Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")