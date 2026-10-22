"""Model training with hyperparameter tuning and MLflow experiment tracking.

Inputs : Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv (from the data-prep job)
Outputs: tourism_project/deployment/best_tourism_model_v1.joblib
         tourism_project/deployment/model_metadata.json
         tourism_project/deployment/tuning_results.csv
"""
import json
import os
import urllib.request
from datetime import datetime, timezone

import joblib
import mlflow
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_VERSION = "v1"
DEPLOY_DIR = os.path.join("tourism_project", "deployment")
MODEL_FILE = os.path.join(DEPLOY_DIR, f"best_tourism_model_{MODEL_VERSION}.joblib")
THRESHOLD = 0.5
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

PARAM_GRID = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.7, 1.0],
}


def set_tracking():
    """Use the MLflow server started by the workflow if it is up, else a local store."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if uri.startswith("http"):
        try:
            urllib.request.urlopen(uri, timeout=3)
        except Exception:
            uri = "file:./mlruns"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("tourism-wellness-package")
    print(f"MLflow tracking URI: {uri}")


def main():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")

    # Class imbalance: about 19% buyers. Weight the positive class.
    scale_pos_weight = float((ytrain == 0).sum() / (ytrain == 1).sum())

    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_FEATURES),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    )
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model_pipeline = make_pipeline(preprocessor, model)

    set_tracking()

    with mlflow.start_run(run_name=f"xgb-gridsearch-{MODEL_VERSION}"):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            model_pipeline, PARAM_GRID, cv=cv, scoring="f1", n_jobs=-1
        )
        grid_search.fit(Xtrain, ytrain)

        # Log every parameter combination as a nested run.
        results = grid_search.cv_results_
        for i in range(len(results["params"])):
            with mlflow.start_run(run_name=f"combo-{i:02d}", nested=True):
                mlflow.log_params(results["params"][i])
                mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
                mlflow.log_metric("std_cv_f1", results["std_test_score"][i])

        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))
        mlflow.log_param("threshold", THRESHOLD)

        metrics = {}
        for split, X, y in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
            proba = best_model.predict_proba(X)[:, 1]
            pred = (proba >= THRESHOLD).astype(int)
            metrics[f"{split}_accuracy"] = accuracy_score(y, pred)
            metrics[f"{split}_precision"] = precision_score(y, pred)
            metrics[f"{split}_recall"] = recall_score(y, pred)
            metrics[f"{split}_f1"] = f1_score(y, pred)
            metrics[f"{split}_roc_auc"] = roc_auc_score(y, proba)
            print(f"\n{split.upper()} classification report")
            print(classification_report(y, pred, digits=4))
        metrics["best_cv_f1"] = grid_search.best_score_
        metrics = {k: round(float(v), 4) for k, v in metrics.items()}
        mlflow.log_metrics(metrics)

        os.makedirs(DEPLOY_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_FILE)
        mlflow.log_artifact(MODEL_FILE, artifact_path="model")

        tuning = pd.DataFrame(results["params"])
        tuning.columns = [c.replace("xgbclassifier__", "") for c in tuning.columns]
        tuning["mean_cv_f1"] = results["mean_test_score"].round(4)
        tuning["std_cv_f1"] = results["std_test_score"].round(4)
        tuning["rank"] = results["rank_test_score"]
        tuning.sort_values("rank").to_csv(
            os.path.join(DEPLOY_DIR, "tuning_results.csv"), index=False
        )

        metadata = {
            "model_version": MODEL_VERSION,
            "model_file": os.path.basename(MODEL_FILE),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "algorithm": "XGBClassifier in sklearn Pipeline",
            "best_params": {k.replace("xgbclassifier__", ""): v
                            for k, v in grid_search.best_params_.items()},
            "scale_pos_weight": round(scale_pos_weight, 4),
            "threshold": THRESHOLD,
            "metrics": metrics,
            "n_train": int(len(Xtrain)),
            "n_test": int(len(Xtest)),
            "library_versions": {
                "scikit-learn": sklearn.__version__,
                "xgboost": xgb.__version__,
                "pandas": pd.__version__,
            },
        }
        with open(os.path.join(DEPLOY_DIR, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    print("\nBest params:", grid_search.best_params_)
    print("Metrics:", json.dumps(metrics, indent=2))
    print(f"Saved best model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
