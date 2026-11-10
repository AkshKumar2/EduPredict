"""EduPredict Web Server — Large-Scale Student Performance Prediction System.

Exclusively serves the Open University Learning Analytics Dataset (OULAD)
trained deep learning and ensemble regression models.

Usage:
  python server.py
Open:
  http://127.0.0.1:5000
"""
import json, sys, threading, webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, send_from_directory
from tensorflow import keras

BASE = Path(__file__).resolve().parent
M = BASE / "models"
if not (M / "meta.json").exists():
    M = BASE / "models_oulad"
O = BASE / "outputs"
if not (O / "results.json").exists():
    O = BASE / "outputs_oulad"
D = BASE / "data"

if not (M / "preprocessor.joblib").exists():
    raise SystemExit(f"Model artifacts not found in {M}. Run 'python train.py' first.")

print(f"Loading trained model artifacts from {M} ...")
meta = json.load(open(M / "meta.json"))
preprocessor = joblib.load(M / "preprocessor.joblib")

# Load neural network
nn_model = None
if (M / "mlp.keras").exists():
    try:
        nn_model = keras.models.load_model(M / "mlp.keras")
        print("✓ Deep Learning MLP model loaded successfully.")
    except Exception as e:
        print(f"Notice: Failed to load Keras model ({e}). Using tree regressors.")

# Load XGBoost if available
xgb_model = None
if (M / "xgboost.joblib").exists():
    try:
        xgb_model = joblib.load(M / "xgboost.joblib")
        print("✓ XGBoost model loaded successfully.")
    except Exception as e:
        print(f"Notice: XGBoost model not loaded ({e}).")

# Load HistGradientBoosting if available
hgb_model = None
if (M / "hgb.joblib").exists():
    try:
        hgb_model = joblib.load(M / "hgb.joblib")
        print("✓ HistGradientBoosting model loaded successfully.")
    except Exception as e:
        pass

COLS = meta["columns"]
CAT_COLS = meta["cat_cols"]
NUM_COLS = meta["num_cols"]
DEFAULTS = meta["defaults"]
SCALE = meta.get("scale", 100)
PASS_MARK = meta.get("pass_mark", 40)
GOOD_MARK = meta.get("good_mark", 70)
TEST_RMSE = float(meta.get("test_rmse", 17.5))
TEST_MAE = float(meta.get("test_mae", 13.5))
SELECTED_MODEL = meta.get("selected_model", "Deep Learning (MLP)")

lock = threading.Lock()
app = Flask(__name__, static_folder="static", static_url_path="/static")


def predict_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Processes tabular features through the fitted preprocessor and returns predictions."""
    df = df.copy()
    for c in COLS:
        if c not in df.columns:
            df[c] = DEFAULTS.get(c, 0)
        elif c in NUM_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(DEFAULTS.get(c, 0))
        else:
            df[c] = df[c].astype(object).where(df[c].notna(), DEFAULTS.get(c, "Unknown"))

    X = preprocessor.transform(df[COLS])

    with lock:
        preds = []
        weights = []

        if xgb_model is not None:
            p_xgb = xgb_model.predict(X)
            preds.append(p_xgb)
            weights.append(0.40)

        if hgb_model is not None:
            p_hgb = hgb_model.predict(X)
            preds.append(p_hgb)
            weights.append(0.30)

        if nn_model is not None:
            p_mlp = nn_model.predict(X, verbose=0).ravel() * SCALE
            preds.append(p_mlp)
            weights.append(0.30 if (xgb_model is not None and hgb_model is not None) else 1.0)

        if len(preds) > 1:
            total_w = sum(weights)
            out = sum(p * (w / total_w) for p, w in zip(preds, weights))
        elif len(preds) == 1:
            out = preds[0]
        else:
            out = np.zeros(len(df))

    return np.clip(out, 0, SCALE)


def calculate_status(score: float) -> str:
    if score < PASS_MARK:
        return "At risk"
    elif score < GOOD_MARK:
        return "Average"
    else:
        return "Good"


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/outputs/<path:name>")
def serve_output(name):
    # Try primary outputs directory first, then fallback
    if (O / name).exists():
        return send_from_directory(O, name)
    fallback = BASE / "outputs_oulad"
    if (fallback / name).exists():
        return send_from_directory(fallback, name)
    return jsonify({"error": f"Asset {name} not found"}), 404


@app.get("/api/health")
def api_health():
    return jsonify({
        "status": "healthy",
        "dataset": "oulad",
        "selected_model": SELECTED_MODEL,
        "features_count": len(COLS),
        "scale": SCALE,
    })


@app.get("/api/meta")
def api_meta():
    keys = [
        "scale", "pass_mark", "good_mark", "title", "lead", "target_label",
        "about_data", "about_model", "sections", "rules", "batch_cols", "dataset",
        "dataset_name", "verified_student_count", "verified_record_count",
        "selected_model", "test_metrics", "test_rmse", "test_mae", "test_r2",
        "raw_total_records", "raw_unique_students"
    ]
    out = {k: meta[k] for k in keys if k in meta}
    out.setdefault("scale", SCALE)
    out.setdefault("pass_mark", PASS_MARK)
    out.setdefault("good_mark", GOOD_MARK)
    out.setdefault("dataset", "oulad")
    out.setdefault("selected_model", SELECTED_MODEL)
    out.setdefault("test_rmse", TEST_RMSE)
    out.setdefault("test_mae", TEST_MAE)
    return jsonify(out)


@app.post("/api/predict")
def api_predict():
    data = request.get_json(force=True)
    score = float(predict_dataframe(pd.DataFrame([data]))[0])

    # Statistically grounded uncertainty interval based on test-set RMSE
    ci_68_low = max(0.0, score - TEST_RMSE)
    ci_68_high = min(SCALE, score + TEST_RMSE)
    ci_95_low = max(0.0, score - 1.96 * TEST_RMSE)
    ci_95_high = min(SCALE, score + 1.96 * TEST_RMSE)

    return jsonify({
        "grade": round(score, 2),
        "status": calculate_status(score),
        "uncertainty": {
            "test_rmse": round(TEST_RMSE, 2),
            "test_mae": round(TEST_MAE, 2),
            "interval_68": [round(ci_68_low, 1), round(ci_68_high, 1)],
            "interval_95": [round(ci_95_low, 1), round(ci_95_high, 1)],
            "stat_confidence": f"±{round(TEST_RMSE, 1)} pts (68% statistical confidence interval based on holdout test RMSE)"
        }
    })


@app.post("/api/batch")
def api_batch():
    try:
        if request.args.get("sample"):
            sample_file = D / "oulad_sample.csv"
            if not sample_file.exists():
                return jsonify({"error": "Sample data file oulad_sample.csv not found"}), 404
            df = pd.read_csv(sample_file)
        else:
            if "file" not in request.files:
                return jsonify({"error": "No file uploaded"}), 400
            file = request.files["file"]
            df = pd.read_csv(file, sep=None, engine="python")
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV file: {e}"}), 400

    df = df.head(3000)
    scores = predict_dataframe(df).round(2)
    df["Predicted_Score"] = scores
    df["Predicted_G3"] = scores  # alias for backwards compatibility
    df["Status"] = df["Predicted_Score"].apply(calculate_status)
    df["Interval_68"] = [
        f"{max(0, s - TEST_RMSE):.1f} – {min(SCALE, s + TEST_RMSE):.1f}"
        for s in scores
    ]

    return jsonify({
        "rows": json.loads(df.to_json(orient="records")),
        "summary": {
            "total_students": len(df),
            "mean_predicted_score": round(float(np.mean(scores)), 2),
            "at_risk_count": int(np.sum(scores < PASS_MARK)),
            "average_count": int(np.sum((scores >= PASS_MARK) & (scores < GOOD_MARK))),
            "good_count": int(np.sum(scores >= GOOD_MARK)),
        }
    })


@app.get("/api/results")
def api_results():
    results_file = O / "results.json"
    if not results_file.exists():
        results_file = BASE / "outputs_oulad" / "results.json"
    if not results_file.exists():
        return jsonify({"error": "Evaluation results not found. Run train.py first."}), 404

    res = json.load(open(results_file))
    available_images = sorted(set(f.name for f in O.glob("*.png")) |
                              set(f.name for f in (BASE / "outputs_oulad").glob("*.png")))
    res["images"] = available_images
    return jsonify(res)


if __name__ == "__main__":
    port = 5000
    print(f"\nStarting EduPredict Web Application on http://127.0.0.1:{port} ...")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
