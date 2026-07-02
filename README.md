#  Containerized Insurance Premium Prediction Engine

An end-to-end, production-style machine learning system that classifies health insurance customers into premium risk categories (**Low, Medium, High**). The project covers the full pipeline — EDA, feature engineering, model training/tuning, a FastAPI inference service, a Streamlit dashboard, and Docker containerization for deployment.

**Timeline:** Jan '26 – Feb '26 | Self Project

##  Objective

To build an optimized classification pipeline that predicts health insurance premium risk categories (Low, Medium, High) from customer demographic, health, and lifestyle data — and serve it through a containerized, production-ready API and web dashboard.

##  Highlights

- **92.64%** test accuracy and **93.20%** 5-fold cross-validation score using an ensemble **Random Forest** classifier, optimized via `RandomizedSearchCV`
- Engineered domain-specific features — **BMI**, **age group**, **lifestyle risk**, and **city tier** — across a **50,000-row** consumer dataset
- Built an async **FastAPI** service with **Pydantic** request validation to catch malformed inputs before they reach the model, reducing inference errors
- **Containerized** the full app ecosystem with **Docker** and published a verified image to **Docker Hub**
- Built an interactive **Streamlit** dashboard that communicates natively with the Dockerized FastAPI backend

##  Machine Learning Pipeline

### 1. Data & EDA
- Source dataset: `insurance_50k-1.csv` (50,000 rows) with fields including age, weight, height, income, smoking status, occupation, and city
- Exploratory analysis on target distribution, smoker vs. premium category, age vs. premium, and income distribution/outliers
- Automated profiling reports generated with `ydata-profiling` (`InsureEDA.html`, `InsureEDANEWFeat.html`)

### 2. Feature Engineering
Raw features are transformed into more predictive, domain-informed features:

| Feature | Description |
|---|---|
| `bmi` | Computed as `weight / height²` |
| `age_group` | Bucketed into `young`, `adult`, `middle_aged`, `senior` |
| `lifestyle_risk` | Derived from smoking status + BMI → `low`, `medium`, `high` |
| `city_tier` | City mapped to `1`, `2`, or `3` based on a curated tier list |

Final model features: `bmi`, `age_group`, `lifestyle_risk`, `city_tier`, `income_lpa`, `occupation`.

### 3. Preprocessing & Modeling
- `ColumnTransformer` with `OneHotEncoder` for categorical features and passthrough for numeric features, wrapped in a scikit-learn `Pipeline`
- **Random Forest Classifier** tuned with `RandomizedSearchCV` (15 iterations, 5-fold CV) across `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`
- Stratified train/test split (80/20) to preserve class balance
- Evaluation via accuracy score, classification report, confusion matrix, and 10-fold cross-validation
- Final pipeline (preprocessing + trained model) serialized with `pickle` as `model.pkl`

##  System Architecture

```
User → Streamlit Dashboard (frontend.py) → FastAPI Backend (app.py) → ML Pipeline (model.pkl) → Prediction
```

- **Frontend:** Streamlit app collects user inputs (age, weight, height, income, smoking status, city, occupation) and posts them to the API
- **Backend:** FastAPI exposes `/`, `/health`, and `/predict` endpoints; validates incoming payloads against a Pydantic schema (`UserInput`) before running inference
- **Model layer:** `predict.py` loads `model.pkl` and exposes a `predict_output` function used by the API
- **Containerization:** Both services run inside Docker, communicating over the container network

## Project Structure

```
Insurance-Premium-Prediction/
├── app.py                     # FastAPI backend (routes: /, /health, /predict)
├── frontend.py                 # Streamlit dashboard
├── Dockerfile                  # Container build definition
├── requirements.txt
├── insurance_50k-1.csv         # Training dataset
├── Insurance_pred.ipynb        # EDA, feature engineering & model training notebook
├── config/
│   └── city_tier.py             # City tier mapping logic
├── model/
│   └── model.pkl                 # Trained, serialized pipeline
├── schema/
│   ├── user_input.py             # Pydantic input schema
│   └── prediction_response.py    # Pydantic response schema
├── predict.py                  # Model loading & inference logic
└── README.md
```

##  Tech Stack

- **Language:** Python 3.13
- **ML:** scikit-learn (RandomForestClassifier, Pipeline, ColumnTransformer, RandomizedSearchCV)
- **Data:** pandas, NumPy
- **EDA:** matplotlib, seaborn, ydata-profiling
- **API:** FastAPI + Pydantic
- **Frontend:** Streamlit
- **Deployment:** Docker, Docker Hub

##  API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Human-readable welcome message |
| `GET` | `/health` | Health check — service status, model version, model load status |
| `POST` | `/predict` | Accepts customer details and returns predicted premium category, confidence, and class probabilities |

**Example request body for `/predict`:**
```json
{
  "age": 30,
  "weight": 65.0,
  "height": 1.7,
  "income_lpa": 10.0,
  "smoker": false,
  "city": "Mumbai",
  "occupation": "private_job"
}
```

**Example response:**
```json
{
  "response": {
    "predicted_category": "Medium",
    "confidence": 0.87,
    "class_probabilities": {
      "Low": 0.08,
      "Medium": 0.87,
      "High": 0.05
    }
  }
}
```

##  Getting Started

### Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt

# Start the FastAPI backend
uvicorn app:app --reload --host 127.0.0.1 --port 8000

# In a separate terminal, start the Streamlit frontend
streamlit run frontend.py
```

### Run with Docker

```bash
# Build the image
docker build -t insurance-premium-predictor .

# Run the container
docker run -p 8000:8000 insurance-premium-predictor

# Or pull the published image from Docker Hub
docker pull <your-dockerhub-username>/insurance-premium-predictor
docker run -p 8000:8000 <your-dockerhub-username>/insurance-premium-predictor
```

##  Results

- **Test Accuracy:** 92.64%
- **5-Fold Cross-Validation Score:** 93.20%
- Full classification report and confusion matrix available in `Insurance_pred.ipynb`

##  Future Improvements

- Add model monitoring/versioning (e.g., MLflow)
- CI/CD pipeline for automated testing and Docker image publishing
- Expand feature set with additional health/lifestyle indicators
- Deploy to a cloud platform (AWS/GCP/Azure) with autoscaling




