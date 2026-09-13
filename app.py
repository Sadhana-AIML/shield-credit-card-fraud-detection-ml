"""
Credit Card Fraud Detection - Backend API
==========================================
LOCAL STORAGE VERSION:
  - Model, scaler, metrics  →  saved in  backend/storage/
  - Prediction history      →  SQLite DB  backend/storage/fraud.db
  - Dataset CSV snapshot    →  backend/storage/dataset.csv
  - All paths are absolute (works from any directory in VS Code)
"""

import os, json, pickle, sqlite3, warnings, datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

# ══════════════════════════════════════════════════════════════════════════════
# LOCAL STORAGE PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))   # .../backend/
STORAGE_DIR  = os.path.join(BASE_DIR, "storage")            # .../backend/storage/
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")     # .../frontend/

# Files stored locally
MODEL_PATH   = os.path.join(STORAGE_DIR, "fraud_model.pkl")
SCALER_PATH  = os.path.join(STORAGE_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(STORAGE_DIR, "metrics.json")
DB_PATH      = os.path.join(STORAGE_DIR, "fraud.db")
DATASET_PATH = os.path.join(STORAGE_DIR, "dataset.csv")

# Create storage folder on startup
os.makedirs(STORAGE_DIR, exist_ok=True)

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


# ══════════════════════════════════════════════════════════════════════════════
# SQLITE — LOCAL DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════
def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT    NOT NULL,
            amount            REAL,
            hour              INTEGER,
            distance          REAL,
            transaction_count INTEGER,
            merchant_risk     REAL,
            velocity_score    REAL,
            prediction        TEXT,
            fraud_prob        REAL,
            risk_level        TEXT,
            is_fraud          INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            algorithm   TEXT,
            accuracy    REAL,
            precision   REAL,
            recall      REAL,
            f1          REAL,
            roc_auc     REAL,
            total_samples INTEGER,
            fraud_samples INTEGER
        )
    """)
    conn.commit()
    conn.close()


def save_prediction_to_db(data: dict, result: dict):
    """Persist a single prediction record to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO predictions
          (timestamp, amount, hour, distance, transaction_count,
           merchant_risk, velocity_score, prediction, fraud_prob, risk_level, is_fraud)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data.get("amount"), data.get("hour"), data.get("distance"),
        data.get("transaction_count"), data.get("merchant_risk"),
        data.get("velocity_score"),
        result["prediction"], result["fraud_prob"],
        result["risk_level"], int(result["is_fraud"])
    ))
    conn.commit()
    conn.close()


def save_training_run_to_db(metrics: dict):
    """Persist a training run record to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO training_runs
          (timestamp, algorithm, accuracy, precision, recall, f1, roc_auc, total_samples, fraud_samples)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        metrics["algorithm"], metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"], metrics["roc_auc"],
        metrics["total_samples"], metrics["fraud_samples"]
    ))
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def generate_dataset(n=5000):
    np.random.seed(42)
    n_legit = n - int(n * 0.02)
    n_fraud = int(n * 0.02)

    legit = {
        "amount":            np.random.exponential(80,  n_legit),
        "hour":              np.random.randint(8, 22,   n_legit),
        "distance":          np.random.exponential(20,  n_legit),
        "transaction_count": np.random.poisson(3,       n_legit),
        "merchant_risk":     np.random.beta(2, 8,       n_legit),
        "velocity_score":    np.random.beta(2, 10,      n_legit),
        "label":             np.zeros(n_legit),
    }
    fraud = {
        "amount":            np.random.exponential(500, n_fraud),
        "hour":              np.random.choice([0,1,2,3,23], n_fraud),
        "distance":          np.random.exponential(500, n_fraud),
        "transaction_count": np.random.poisson(15,      n_fraud),
        "merchant_risk":     np.random.beta(8, 2,       n_fraud),
        "velocity_score":    np.random.beta(8, 2,       n_fraud),
        "label":             np.ones(n_fraud),
    }

    df = pd.concat(
        [pd.DataFrame(legit), pd.DataFrame(fraud)],
        ignore_index=True
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save dataset snapshot locally
    df.to_csv(DATASET_PATH, index=False)
    print(f"  Dataset saved → {DATASET_PATH}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING  (saves to local storage/)
# ══════════════════════════════════════════════════════════════════════════════
def train_model(algorithm="random_forest"):
    df = generate_dataset()
    X  = df.drop("label", axis=1)
    y  = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model_map = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "logistic":      LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    }
    model = model_map.get(algorithm, model_map["random_forest"])
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    cm     = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy":         round(accuracy_score(y_test, y_pred)                   * 100, 2),
        "precision":        round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall":           round(recall_score(y_test, y_pred, zero_division=0)    * 100, 2),
        "f1":               round(f1_score(y_test, y_pred, zero_division=0)        * 100, 2),
        "roc_auc":          round(roc_auc_score(y_test, y_prob)                    * 100, 2),
        "confusion_matrix": cm,
        "algorithm":        algorithm,
        "total_samples":    len(df),
        "fraud_samples":    int(df["label"].sum()),
        "legit_samples":    int((df["label"] == 0).sum()),
    }

    # ── Save to local storage/ ────────────────────────────────────────────────
    with open(MODEL_PATH,   "wb") as f: pickle.dump(model,   f)
    with open(SCALER_PATH,  "wb") as f: pickle.dump(scaler,  f)
    with open(METRICS_PATH, "w")  as f: json.dump(metrics,   f, indent=2)

    # ── Save training run to SQLite ───────────────────────────────────────────
    save_training_run_to_db(metrics)

    print(f"  Model   saved → {MODEL_PATH}")
    print(f"  Scaler  saved → {SCALER_PATH}")
    print(f"  Metrics saved → {METRICS_PATH}")
    print(f"  DB      saved → {DB_PATH}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_model_scaler():
    with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    return model, scaler

def build_features(tx: dict) -> np.ndarray:
    return np.array([[
        float(tx.get("amount",            0)),
        float(tx.get("hour",              12)),
        float(tx.get("distance",          0)),
        float(tx.get("transaction_count", 1)),
        float(tx.get("merchant_risk",     0.1)),
        float(tx.get("velocity_score",    0.1)),
    ]])

def get_storage_info():
    """Return sizes of all local storage files."""
    files = {
        "fraud_model.pkl": MODEL_PATH,
        "scaler.pkl":      SCALER_PATH,
        "metrics.json":    METRICS_PATH,
        "fraud.db":        DB_PATH,
        "dataset.csv":     DATASET_PATH,
    }
    info = {}
    for name, path in files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            info[name] = f"{size:,} bytes" if size < 1024*1024 else f"{size/1024/1024:.2f} MB"
        else:
            info[name] = "not created yet"
    return info


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status":        "online",
        "model_trained": os.path.exists(MODEL_PATH),
        "storage_dir":   STORAGE_DIR,
        "storage_files": get_storage_info(),
    })


@app.route("/api/train", methods=["POST"])
def api_train():
    data = request.get_json(force=True, silent=True) or {}
    algo = data.get("algorithm", "random_forest")
    if algo not in {"random_forest", "logistic", "decision_tree"}:
        return jsonify({"error": f"Unknown algorithm: {algo}"}), 400
    try:
        metrics = train_model(algo)
        return jsonify({"status": "success", "metrics": metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "Model not trained yet"}), 404
    with open(METRICS_PATH) as f:
        return jsonify(json.load(f))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Train the model first"}), 400
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Empty or invalid JSON body"}), 400
    try:
        model, scaler = load_model_scaler()
        scaled = scaler.transform(build_features(data))
        pred   = model.predict(scaled)[0]
        prob   = model.predict_proba(scaled)[0]
        risk   = round(float(prob[1]) * 100, 2)

        result = {
            "prediction": "FRAUD" if int(pred) == 1 else "LEGITIMATE",
            "fraud_prob": risk,
            "legit_prob": round(float(prob[0]) * 100, 2),
            "risk_level": "HIGH" if risk >= 70 else ("MEDIUM" if risk >= 40 else "LOW"),
            "is_fraud":   bool(int(pred) == 1),
        }

        # Save prediction to local SQLite
        save_prediction_to_db(data, result)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch", methods=["POST"])
def api_batch():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Train the model first"}), 400
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Empty or invalid JSON body"}), 400
    transactions = data.get("transactions", [])
    if not transactions:
        return jsonify({"error": "No transactions provided"}), 400
    try:
        model, scaler = load_model_scaler()
        results = []
        for i, tx in enumerate(transactions):
            scaled = scaler.transform(build_features(tx))
            pred   = model.predict(scaled)[0]
            prob   = model.predict_proba(scaled)[0]
            risk   = round(float(prob[1]) * 100, 2)
            r = {
                "id":         i + 1,
                "amount":     tx.get("amount"),
                "prediction": "FRAUD" if int(pred) == 1 else "LEGITIMATE",
                "risk_score": risk,
                "is_fraud":   bool(int(pred) == 1),
            }
            results.append(r)
            # Save each to DB
            save_prediction_to_db(tx, {
                "prediction": r["prediction"], "fraud_prob": risk,
                "risk_level": "HIGH" if risk>=70 else ("MEDIUM" if risk>=40 else "LOW"),
                "is_fraud": r["is_fraud"]
            })

        fraud_count = sum(1 for r in results if r["is_fraud"])
        return jsonify({"total": len(results), "fraud_detected": fraud_count, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    """Return last 50 predictions from local SQLite DB."""
    if not os.path.exists(DB_PATH):
        return jsonify({"history": []})
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT 50"
        ).fetchall()
        conn.close()
        return jsonify({"history": [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/clear", methods=["DELETE"])
def api_history_clear():
    """Clear all prediction history from local SQLite DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        return jsonify({"status": "cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/training-runs", methods=["GET"])
def api_training_runs():
    """Return all training run history from local SQLite DB."""
    if not os.path.exists(DB_PATH):
        return jsonify({"runs": []})
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM training_runs ORDER BY id DESC"
        ).fetchall()
        conn.close()
        return jsonify({"runs": [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/storage", methods=["GET"])
def api_storage():
    """Show what is saved in local storage/ folder."""
    return jsonify({
        "storage_dir":   STORAGE_DIR,
        "files":         get_storage_info(),
        "model_trained": os.path.exists(MODEL_PATH),
    })


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  🛡️  Credit Card Fraud Detection — LOCAL STORAGE MODE")
    print("=" * 60)
    print(f"  Storage folder : {STORAGE_DIR}")
    print(f"  Frontend folder: {os.path.abspath(FRONTEND_DIR)}")
    print(f"  API URL        : http://localhost:5000")
    print("=" * 60)

    # Init SQLite DB
    init_db()
    print(f"\n  ✅  SQLite DB ready  → {DB_PATH}")

    # Auto-train if no model saved yet
    if not os.path.exists(MODEL_PATH):
        print("\n  ⚙️   No model in storage — auto-training Random Forest...")
        try:
            m = train_model("random_forest")
            print(f"  ✅  Model saved!  Accuracy={m['accuracy']}%  AUC={m['roc_auc']}%")
        except Exception as e:
            print(f"  ❌  Auto-train failed: {e}")
    else:
        print(f"\n  ✅  Model loaded from storage → {MODEL_PATH}")

    print("\n  📂  Local storage contents:")
    for name, size in get_storage_info().items():
        print(f"      {name:<22} {size}")

    print("\n  🌐  Open frontend/index.html in your browser\n")
    app.run(debug=True, port=5000, use_reloader=False)
