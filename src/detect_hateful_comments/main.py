
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from groq_classifier import predict_groq
import numpy as np

# Import your ensemble function
from ensemble import predict_ensemble

app = FastAPI(
    title="Toxicity Classification API",
    description="Multi-label toxicity classification using a soft voting ensemble and gpt-oss-120b",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    comment: str = Field(..., min_length=1)


@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "Toxicity Classification API is running"
    }


@app.post("/predict-ensemble")
def predict(request: PredictionRequest):

    try:
        prediction_result = predict_ensemble(
            request.comment
        )

        result_df = pd.DataFrame({
            "label": prediction_result["labels"],
            "probability": (
                prediction_result["probabilities"][0] * 100
            ).round(2)
        })

        # Convert DataFrame to JSON-compatible dictionary
        predictions = result_df.to_dict(
            orient="records"
        )

        return {
            "comment": request.comment,
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/predict-llm")
def predict_llm(request: PredictionRequest):

    try:
        return predict_groq(request.comment)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/predict-smart")
def predict_smart(request: PredictionRequest):

    try:
        # 1. Run the ML ensemble first
        ml_result = predict_ensemble(request.comment)

        labels = ml_result["labels"]
        probabilities = np.asarray(
            ml_result["probabilities"][0],
            dtype=float
        )

        # 2. Build the ensemble predictions
        ensemble_predictions = [
            {
                "label": label,
                "probability": round(float(prob) * 100, 2)
            }
            for label, prob in zip(labels, probabilities)
        ]

        # 3. Evaluate uncertainty
        # Probabilities close to 0.5 are considered uncertain.
        classification_threshold = 0.5
        uncertainty_margin = 0.15

        uncertain_labels = [
            label
            for label, probability in zip(labels, probabilities)
            if abs(probability - classification_threshold)
            <= uncertainty_margin
        ]

        # 4. If the ensemble is confident, return its result
        if not uncertain_labels:
            return {
                "comment": request.comment,
                "source": "ensemble",
                "llm_called": False,
                "predictions": ensemble_predictions
            }

        # 5. Otherwise, fall back to the LLM
        try:
            llm_result = predict_groq(request.comment)

            return {
                "comment": request.comment,
                "source": "llm_fallback",
                "llm_called": True,
                "uncertain_labels": uncertain_labels,
                "ensemble_predictions": ensemble_predictions,
                "predictions": llm_result["predictions"]
            }

        except Exception as e:
            return {
                "comment": request.comment,
                "source": "ensemble_fallback",
                "llm_called": True,
                "llm_error": str(e),
                "predictions": ensemble_predictions
            }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )