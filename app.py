
import os
import re
import json
import math
import traceback
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "movie-success-project-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

DATA_PATH = os.path.join(DATA_DIR, "movies.csv")
ARTIFACT_PATH = os.path.join(MODEL_DIR, "artifacts.json")

state = {
    "trained": False,
    "message": "Upload your movie CSV to train the model.",
    "metrics": {},
    "columns": [],
    "preview": [],
    "charts": {},
    "feature_importance": [],
    "model_name": "Gradient Boosting Classifier"
}

model_bundle = {"pipeline": None, "feature_columns": [], "threshold": 8.0, "numeric_columns": []}

def normalize_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")

def find_col(df, candidates):
    normalized = {normalize_col(c): c for c in df.columns}
    for cand in candidates:
        if cand in normalized:
            return normalized[cand]
    for c in df.columns:
        nc = normalize_col(c)
        for cand in candidates:
            if cand in nc:
                return c
    return None

def numeric_series(df, col):
    if col is None:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(
        df[col].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False),
        errors="coerce"
    )

def detect_columns(df):
    return {
        "title": find_col(df, ["series_title", "title", "movie_title", "name"]),
        "year": find_col(df, ["released_year", "release_year", "year"]),
        "runtime": find_col(df, ["runtime", "duration"]),
        "votes": find_col(df, ["no_of_votes", "votes", "vote_count", "number_of_votes"]),
        "gross": find_col(df, ["gross", "box_office", "revenue"]),
        "rating": find_col(df, ["imdb_rating", "rating", "score"]),
        "genre": find_col(df, ["genre", "genres"]),
        "director": find_col(df, ["director"]),
        "overview": find_col(df, ["overview", "description", "plot"]),
        "certificate": find_col(df, ["certificate", "certification", "rating_certificate"])
    }

def load_df():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

def build_training_frame(df):
    cols = detect_columns(df)
    required = [cols["year"], cols["runtime"], cols["votes"], cols["gross"], cols["rating"]]
    if any(x is None for x in required):
        missing = []
        labels = {"year":"Release Year", "runtime":"Runtime", "votes":"Votes", "gross":"Gross", "rating":"IMDb Rating"}
        for k, c in cols.items():
            if k in labels and c is None:
                missing.append(labels[k])
        raise ValueError("Required columns not found: " + ", ".join(missing))

    out = pd.DataFrame(index=df.index)
    out["release_year"] = numeric_series(df, cols["year"])
    out["runtime"] = numeric_series(df, cols["runtime"])
    out["votes"] = numeric_series(df, cols["votes"])
    out["gross"] = numeric_series(df, cols["gross"])
    out["success"] = (numeric_series(df, cols["rating"]) >= 8.0).astype(int)

    # Log-transform high-skew count/revenue variables.
    out["log_votes"] = np.log1p(out["votes"].clip(lower=0))
    out["log_gross"] = np.log1p(out["gross"].clip(lower=0))

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["success"])
    return out, cols

def train_model():
    global model_bundle, state

    df = load_df()
    if df is None or len(df) < 30:
        raise ValueError("Please upload a CSV dataset with at least 30 movie records.")

    train_df, cols = build_training_frame(df)
    feature_columns = ["release_year", "runtime", "log_votes", "log_gross"]
    X = train_df[feature_columns]
    y = train_df["success"]

    if y.nunique() < 2:
        raise ValueError("The target contains only one class. The dataset needs both successful and non-successful movies.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=5,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)) * 100, 2),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)) * 100, 2),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)) * 100, 2),
        "auc": round(float(roc_auc_score(y_test, proba)) * 100, 2) if len(np.unique(y_test)) == 2 else None,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "total": int(len(train_df)),
        "successful": int(y.sum()),
        "not_successful": int((y == 0).sum()),
        "confusion_matrix": cm.tolist()
    }

    importances = pipeline.named_steps["model"].feature_importances_
    pretty = {
        "release_year": "Release Year",
        "runtime": "Runtime",
        "log_votes": "Audience Votes",
        "log_gross": "Gross Revenue"
    }
    feature_importance = [
        {"name": pretty[f], "value": round(float(v) * 100, 2)}
        for f, v in zip(feature_columns, importances)
    ]
    feature_importance.sort(key=lambda x: x["value"], reverse=True)

    title_col = cols["title"]
    titles = df[title_col].fillna("Unknown").astype(str).tolist() if title_col else [f"Movie {i+1}" for i in range(len(df))]
    ratings = numeric_series(df, cols["rating"]).fillna(0)
    years = numeric_series(df, cols["year"]).fillna(0)

    top_movies = []
    temp = pd.DataFrame({
        "title": titles,
        "rating": ratings,
        "year": years
    }).sort_values("rating", ascending=False).head(10)
    for _, row in temp.iterrows():
        top_movies.append({
            "title": str(row["title"]),
            "rating": round(float(row["rating"]), 1),
            "year": int(row["year"]) if row["year"] else None
        })

    genre_counts = {}
    if cols["genre"]:
        for raw in df[cols["genre"]].fillna("").astype(str):
            for g in raw.split(","):
                g = g.strip()
                if g:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    year_counts = {}
    for yv in years:
        if yv:
            yv = int(yv)
            year_counts[yv] = year_counts.get(yv, 0) + 1
    year_items = sorted(year_counts.items())

    model_bundle = {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "threshold": 8.0,
        "numeric_columns": feature_columns
    }

    state = {
        "trained": True,
        "message": "Model trained successfully.",
        "metrics": metrics,
        "columns": list(df.columns),
        "preview": df.head(8).fillna("").to_dict(orient="records"),
        "charts": {
            "genres": [{"label": k, "value": v} for k, v in top_genres],
            "years": [{"label": k, "value": v} for k, v in year_items],
            "top_movies": top_movies
        },
        "feature_importance": feature_importance,
        "model_name": "Gradient Boosting Classifier"
    }

    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "feature_importance": feature_importance,
            "model_name": state["model_name"],
            "columns": state["columns"]
        }, f, indent=2)

    return state

@app.route("/")
def index():
    return render_template("index.html", state=state)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", state=state)

@app.route("/predict", methods=["POST"])
def predict():
    if not state["trained"] or model_bundle["pipeline"] is None:
        flash("Train the model first by uploading your CSV dataset.", "warning")
        return redirect(url_for("index"))

    try:
        movie_name = request.form.get("movie_name", "Untitled Movie")
        year = float(request.form.get("year", 0))
        runtime = float(request.form.get("runtime", 0))
        votes = float(request.form.get("votes", 0))
        gross = float(request.form.get("gross", 0))

        if year < 1888 or runtime <= 0 or votes < 0 or gross < 0:
            raise ValueError("Please enter valid movie values.")

        X_new = pd.DataFrame([{
            "release_year": year,
            "runtime": runtime,
            "log_votes": math.log1p(votes),
            "log_gross": math.log1p(gross)
        }])

        probability = float(model_bundle["pipeline"].predict_proba(X_new)[0, 1])
        prediction = int(probability >= 0.5)

        result = {
            "movie_name": movie_name,
            "probability": round(probability * 100, 2),
            "prediction": prediction,
            "label": "SUCCESSFUL MOVIE" if prediction else "NOT SUCCESSFUL MOVIE",
            "year": int(year),
            "runtime": int(runtime),
            "votes": int(votes),
            "gross": gross
        }

        return render_template("result.html", state=state, result=result)
    except Exception as e:
        flash(str(e), "danger")
        return redirect(url_for("index"))

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not state["trained"] or model_bundle["pipeline"] is None:
        return jsonify({"error": "Model is not trained. Upload a dataset first."}), 400
    try:
        payload = request.get_json(force=True)
        X_new = pd.DataFrame([{
            "release_year": float(payload["year"]),
            "runtime": float(payload["runtime"]),
            "log_votes": math.log1p(float(payload["votes"])),
            "log_gross": math.log1p(float(payload["gross"]))
        }])
        p = float(model_bundle["pipeline"].predict_proba(X_new)[0, 1])
        return jsonify({
            "success_probability": round(p * 100, 2),
            "prediction": "SUCCESSFUL MOVIE" if p >= 0.5 else "NOT SUCCESSFUL MOVIE"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("dataset")
    if not file or not file.filename:
        flash("Please select a CSV file.", "warning")
        return redirect(url_for("index"))
    if not file.filename.lower().endswith(".csv"):
        flash("Only CSV files are supported.", "danger")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)
    shutil_path = DATA_PATH
    import shutil
    shutil.copy2(save_path, shutil_path)

    try:
        train_model()
        flash("Dataset uploaded and model trained successfully.", "success")
    except Exception as e:
        flash(f"Training failed: {e}", "danger")
    return redirect(url_for("dashboard"))

@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        train_model()
        flash("Model retrained successfully.", "success")
    except Exception as e:
        flash(f"Retraining failed: {e}", "danger")
    return redirect(url_for("dashboard"))

@app.route("/about")
def about():
    return render_template("about.html", state=state)

@app.route("/api/status")
def api_status():
    return jsonify({
        "trained": state["trained"],
        "message": state["message"],
        "model": state["model_name"],
        "metrics": state["metrics"]
    })

if __name__ == "__main__":
    # If a dataset is already placed in data/movies.csv, train automatically.
    if os.path.exists(DATA_PATH):
        try:
            train_model()
        except Exception as e:
            state["message"] = f"Dataset found, but training is waiting: {e}"
    app.run(debug=True, host="127.0.0.1", port=5000)
