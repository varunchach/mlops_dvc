# monitor/poll_monitor.py
import time
import mlflow

MLFLOW_EXP = "mlops_demo_experiment"
METRIC_NAME = "accuracy_custom_eval"
THRESHOLD = 0.80
POLL_INTERVAL = 15  # seconds for demo

def get_latest_metric(exp_name):
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        return None, None
    df = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=10)
    if df.empty:
        return None, None
    latest = df.iloc[0]
    acc = latest.get(f"metrics.{METRIC_NAME}")
    run_id = latest["run_id"]
    return (float(acc) if acc is not None else None), run_id

def monitor_loop():
    print("Starting MLflow poll monitor. Poll every", POLL_INTERVAL, "s.")
    while True:
        acc, run_id = get_latest_metric(MLFLOW_EXP)
        if acc is None:
            print("No runs yet.")
        else:
            print(f"Latest run {run_id} -> {METRIC_NAME} = {acc}")
            if acc < THRESHOLD:
                print(f"ALERT: run {run_id} metric {METRIC_NAME} = {acc} below THRESHOLD {THRESHOLD}")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    monitor_loop()
