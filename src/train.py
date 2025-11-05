# src/train.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import read_params
from datetime import datetime


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("mlops_experiment")


def run(params_path="params.yaml", run_name=None, tags=None, extra_note=None):
    cfg = read_params(params_path)
    dp = cfg["data"]
    tp = cfg["train"]

    # enable autologging if requested (must be before fit())
    autolog_flag = bool(tp.get("autolog", False))
    if autolog_flag:
        mlflow.sklearn.autolog()
        print("MLflow autologging for sklearn ENABLED")
    else:
        print("MLflow autologging for sklearn DISABLED (manual logging will be used)")

    # Load data
    df = pd.read_csv(dp["raw_path"])
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=tp["test_size"], random_state=tp["random_state"]
    )

    # Set experiment
    exp_name = tp.get("experiment_name", "default")
    mlflow.set_experiment(exp_name)

    # Start run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # manual params logging only when autolog is off (autolog will log estimator params)
        if not autolog_flag:
            mlflow.log_params({k: v for k, v in tp.items() if k != "experiment_name"})

        # Fit model
        model = LogisticRegression(max_iter=tp["max_iter"], C=tp["C"], solver=tp.get("solver","lbfgs"))
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="macro"))

        # Always log custom business metrics (distinct names to avoid confusion)
        mlflow.log_metric("accuracy_custom_eval", acc)
        mlflow.log_metric("f1_macro_custom_eval", f1)

        # Add tags and notes
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)
        if extra_note:
            mlflow.log_text(extra_note, "notes.txt")

        # Save model artifact explicitly
        os.makedirs("models", exist_ok=True)
        model_path = f"models/logreg_{run_id}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        # Also log via mlflow.sklearn for serving compatibility (autolog may already do this)
        mlflow.sklearn.log_model(model, artifact_path="skmodel")

        print(f"Run {run_id} finished. accuracy={acc:.4f}, f1_macro={f1:.4f}")
        return run.info

if __name__ == "__main__":
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_name = f"demo_run_{ts}"
    tags = {"stage": "demo", "owner": "you"}
    run_info = run(run_name=run_name, tags=tags, extra_note="Autolog demo")
    print("Run info:", run_info)
