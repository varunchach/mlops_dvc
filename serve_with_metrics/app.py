# serve_with_metrics/app.py
from flask import Flask, request, jsonify
import mlflow.sklearn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

################################## Replace <RUN_ID> with an actual run id after running training
MODEL_URI = "runs:/e7060aa0a99d48b78bbae0e617991917/skmodel"  

app = Flask(__name__)
model = mlflow.sklearn.load_model(MODEL_URI)

REQUEST_COUNT = Counter('predict_requests_total', 'Total prediction requests')
REQUEST_LATENCY = Histogram('predict_request_latency_seconds', 'Prediction latency seconds')

@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    REQUEST_COUNT.inc()
    payload = request.get_json()
    inputs = payload.get("inputs")
    preds = model.predict(inputs).tolist()
    latency = time.time() - start
    REQUEST_LATENCY.observe(latency)
    return jsonify({"predictions": preds, "latency": latency})

@app.route("/metrics")
def metrics():
    resp = generate_latest()
    return resp, 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(port=5005)
