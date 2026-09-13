# Shield Credit Card Fraud Detection Using ML
### College Project | Python + Flask + scikit-learn + HTML/CSS/JS

---

## Quick Start (3 Steps)

### Step 1 — Install Python packages
Open a terminal inside VS Code (Ctrl + `) and run:
```bash
py -m pip install flask flask-cors scikit-learn numpy pandas
```

### Step 2 — Run the backend
```bash
cd backend
py app.py
```
You will see:
```
=======================================================
  Shield  Credit Card Fraud Detection API
=======================================================
  URL : http://localhost:5000
  Auto-training Random Forest model...
  Model ready!  Accuracy=100.0%  AUC=100.0%

  Open frontend/index.html in your browser.
```

### Step 3 — Open the frontend
Double-click `frontend/index.html` to open it in your browser.
Or use the **Live Server** VS Code extension and click "Go Live".

---

## Project Structure
```
fraud-detection/
 backend/
   app.py              Flask REST API + ML models (FIXED)
   requirements.txt    Python dependencies
 frontend/
   index.html          Full dashboard UI (no build tools needed)
 README.md
```

---

## Bugs Fixed in This Version

| Bug | Fix Applied |
|-----|-------------|
| Paths broke when running from wrong directory | All paths now use `os.path.abspath(__file__)` |
| `send_from_directory` crash | Now uses absolute FRONTEND_DIR path |
| `precision_score` division-by-zero crash | Added `zero_division=0` parameter |
| `request.get_json()` returned None and crashed | Added `silent=True` + `or {}` guard |
| No health-check endpoint | Added `/api/status` endpoint |
| Frontend checked wrong endpoint for API status | Frontend now uses `/api/status` |

---

## How to Use

| Tab | What it does |
|-----|-------------|
| Train Model | Choose algorithm (Random Forest / Logistic / Decision Tree) and train |
| Predict | Enter a single transaction, get FRAUD / LEGITIMATE verdict + risk score |
| Batch Test | Paste JSON array of transactions, analyze all at once |
| Metrics | View Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| About | Project info, feature table, API reference |

---

## REST API Reference

### Health Check
```
GET http://localhost:5000/api/status
```

### Train Model
```
POST http://localhost:5000/api/train
{"algorithm": "random_forest"}    # or "logistic" or "decision_tree"
```

### Single Prediction
```
POST http://localhost:5000/api/predict
{
  "amount": 150.0,
  "hour": 14,
  "distance": 10,
  "transaction_count": 3,
  "merchant_risk": 0.10,
  "velocity_score": 0.10
}
```

### Batch Predictions
```
POST http://localhost:5000/api/batch
{
  "transactions": [
    {"amount":50,   "hour":14, "distance":5,   "transaction_count":2,  "merchant_risk":0.08, "velocity_score":0.07},
    {"amount":1200, "hour":2,  "distance":800, "transaction_count":15, "merchant_risk":0.92, "velocity_score":0.95}
  ]
}
```

### Get Metrics
```
GET http://localhost:5000/api/metrics
```

---

## ML Algorithms

| Algorithm | Notes |
|-----------|-------|
| Random Forest | Best for imbalanced fraud data. Uses 100 trees. |
| Logistic Regression | Fast linear baseline. Good interpretability. |
| Decision Tree | Easiest to explain. Max depth = 8. |

---

## Features Used

| Feature | Description |
|---------|-------------|
| amount | Transaction amount (USD) |
| hour | Hour of day 0–23 |
| distance | Distance from cardholder home (km) |
| transaction_count | Number of transactions in last 24 hours |
| merchant_risk | Merchant risk score 0.0–1.0 |
| velocity_score | Spending velocity score 0.0–1.0 |

---

## Requirements

- Python 3.8 or newer
- pip
- VS Code (recommended)
- Chrome, Firefox, or Edge browser

---

*Built for college-level ML coursework. Runs 100% locally. No cloud or paid APIs needed.*
