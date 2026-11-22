# AI-Powered Toxicity Classification API

A multi-label toxicity classification API built with **FastAPI**, combining a Machine Learning ensemble with an optional LLM fallback.

The project explores a hybrid approach to content moderation: using traditional Machine Learning for initial classification and involving an LLM only when the ensemble's predictions indicate uncertainty.

## Architecture

The API provides three prediction strategies:

1. **Ensemble Classification:** Uses a soft voting ensemble to predict toxicity categories.
2. **LLM Classification:** Sends comments directly to an LLM for toxicity analysis.
3. **Smart Classification:** Combines both approaches using confidence-based routing.

## How It Works

### 1. Machine Learning Ensemble

The ensemble performs multi-label classification and predicts probabilities for five toxicity categories:

- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

Each category receives an individual probability score.

### 2. Confidence-Based Routing

Instead of calling an LLM for every comment, the smart prediction endpoint evaluates the ensemble's predicted probabilities.

The current implementation uses:

| Parameter                |   Value |
| ------------------------ | ------: |
| Classification threshold |     50% |
| Uncertainty margin       |     15% |
| Uncertainty range        | 35%–65% |

For each predicted label, the system checks whether its probability falls within the uncertainty range.

- **Below 35%:** Confident negative prediction.
- **Above 65%:** Confident positive prediction.
- **Between 35% and 65%:** Uncertain prediction.

If all labels fall outside the uncertainty range, the ensemble predictions are returned directly.

If at least one label falls within the uncertainty range, the comment is sent to the LLM for additional analysis.

> The uncertainty range is a configurable routing heuristic based on predicted probabilities, not a guarantee of statistically calibrated confidence.

### 3. LLM Fallback

If the LLM request fails, the API falls back to the original ensemble predictions instead of failing the entire request.

This keeps the ensemble as the primary classification layer while making LLM analysis an optional additional layer.

## API Endpoints

All POST endpoints accept the following request structure:

```json
{
  "comment": "Your comment here"
}
```

### Health Check

`GET /`

Checks whether the API is running.

### Ensemble Prediction

`POST /predict-ensemble`

Runs only the Machine Learning ensemble.

Example response:

```json
{
  "comment": "This is the worst product I've ever bought.",
  "predictions": [
    {
      "label": "severe_toxic",
      "probability": 1.2
    },
    {
      "label": "insult",
      "probability": 10.55
    }
  ]
}
```

### Direct LLM Prediction

`POST /predict-llm`

Sends the comment directly to the LLM classifier without running the ensemble.

### Smart Prediction

`POST /predict-smart`

Runs the ensemble first and determines whether an LLM call is necessary.

**Possible response: Ensemble classification**

```json
{
  "comment": "This is the worst product I've ever bought.",
  "source": "ensemble",
  "llm_called": false,
  "predictions": []
}
```

**Possible response: LLM fallback**

```json
{
  "comment": "Example comment",
  "source": "llm_fallback",
  "llm_called": true,
  "uncertain_labels": ["insult"],
  "ensemble_predictions": [],
  "predictions": []
}
```

**Possible response: LLM failure**

```json
{
  "comment": "Example comment",
  "source": "ensemble_fallback",
  "llm_called": true,
  "llm_error": "LLM request failed",
  "predictions": []
}
```

The examples above illustrate the response structures. Actual predictions depend on the input and underlying models.

## Project Structure

```text
.
├── main.py
├── ensemble.py
├── groq_classifier.py
├── requirements.txt
└── ...
```

| File                 | Responsibility                        |
| -------------------- | ------------------------------------- |
| `main.py`            | FastAPI application and API endpoints |
| `ensemble.py`        | Machine Learning ensemble inference   |
| `groq_classifier.py` | LLM-based toxicity classification     |
| `requirements.txt`   | Python dependencies                   |

Additional model artifacts and configuration files may be required by the ensemble implementation.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/omarzahir/toxic-comments-classifier.git
cd toxic-comments-classifier
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**

```powershell
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

If you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the basic API dependencies:

```bash
pip install fastapi uvicorn pandas numpy
```

Make sure to install any additional dependencies required by your ensemble and LLM implementations.

### 4. Configure Environment Variables

Configure the API credentials required by `groq_classifier.py`.

For example, if your implementation uses `GROQ_API_KEY`:

**Windows PowerShell:**

```powershell
$env:GROQ_API_KEY="your_api_key"
```

**Linux / macOS:**

```bash
export GROQ_API_KEY="your_api_key"
```

Never commit API keys or other sensitive credentials to GitHub.

### 5. Run the API

```bash
uvicorn main:app --reload
```

The API will be available at:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

**Note:** The API requires these files to be present locally before running inference.

## Trained Models

The trained model files are hosted separately on Hugging Face due to their combined size (327 MB) and are not included in this GitHub repository.

The following files are required to run the inference API:

| File                      | Description                               |
| ------------------------- | ----------------------------------------- |
| `logistic_regression.pkl` | Trained Logistic Regression model         |
| `nn.pkl`                  | Trained Neural Network model              |
| `svc.pkl`                 | Trained Support Vector Classifier model   |
| `target_columns.pkl`      | Target label columns used during training |

### Download and Setup

1. Download all model files from the [Hugging Face Model Repository](https://huggingface.co/YOUR_USERNAME/toxic-comment-classifier-models).

2. Create a `models/` directory inside `src/detect_hateful_comments/`.

3. Place all downloaded `.pkl` files inside the newly created directory.

Your project structure should look like this:

```text
src/
└── detect_hateful_comments/
    ├── models/
    │   ├── logistic_regression.pkl
    │   ├── nn.pkl
    │   ├── svc.pkl
    │   └── target_columns.pkl
    ├── main.py
    ├── ensemble.py
    └── groq_classifier.py
```

**Note:** The `models/` directory is excluded from GitHub due to its size. Make sure to download and place the required files before running the inference API.

## Future Improvements

Potential areas for further experimentation:

- Fine-tuning classification thresholds using validation data.
- Improving probability calibration.
- Evaluating ensemble performance across different toxicity categories.
- Measuring the trade-off between classification quality, latency, and LLM API usage.
- Incorporating human feedback into future model improvements.

## Technologies

- Python
- FastAPI
- Scikit-learn / Machine Learning
- NumPy
- Pandas
- Groq API / LLM

---

**Note:** This repository contains the classification API layer. Ensure that the required trained model artifacts, inference modules, dependencies, and API configuration are available before running the application.
