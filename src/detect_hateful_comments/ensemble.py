
import joblib
import tensorflow as tf

# Load models once when the application starts
logistic_model_pkl = joblib.load(
    "models/logistic_regression.pkl"
)

svc_model_pkl = joblib.load(
    "models/svc.pkl"
)

nn_model_pkl = joblib.load(
    "models/nn.pkl"
)

target_columns = joblib.load(
    "models/target_columns.pkl"
)

def predict_ensemble(comments):

    if isinstance(comments, str):
        comments = [comments]

    logistic_probs = logistic_model_pkl.predict_proba(
        comments
    )

    svc_probs = svc_model_pkl.predict_proba(
        comments
    )

    nn_probs = nn_model_pkl.predict_proba(
        comments,
        verbose=0
    )

    ensemble_probs = (
        logistic_probs + svc_probs + nn_probs
    ) / 3

    return {
        "probabilities": ensemble_probs,
        "labels": target_columns
    }
