# 🧠 MLOps — MLflow Hands-On Demo  

## 🚀 Overview  
This repository is a **beginner-friendly MLflow demo** designed to help learners understand how experiment tracking, metric logging, and artifact management work in an MLOps setup.  

It includes:  
- a simple training script (`src/train.py`)  
- configuration file (`params.yaml`)  
- demo data (`data/`)  
- an inline notebook (`mlflow_inline_demo.ipynb`) to visualize and understand MLflow runs.  

---

## ⚙️ Prerequisites  
- Python 3.8 or higher  
- Git installed  
- MLflow library  
- Optional: Jupyter Notebook (for running the `.ipynb` demo)

### 🧩 Setup Environment  
```bash
python -m venv venv
.\venv\Scripts\activate        # for Windows
# source venv/bin/activate     # for Linux/Mac
pip install --upgrade pip
pip install -r requirements.txt
````

If `requirements.txt` isn’t present, install the basics manually:

```bash
pip install mlflow scikit-learn pandas matplotlib pyyaml
```

---

## 📂 Repository Structure

```
mlops_mlflow/
├─ data/                      # sample dataset folder
│  └─ raw/                    # contains demo CSV for training
├─ src/
│  └─ train.py                # training script that logs experiments to MLflow
├─ mlflow_inline_demo.ipynb   # notebook version for interactive exploration
├─ params.yaml                # stores model hyperparameters and config
├─ .gitignore
└─ README.md
```

---

## 🧠 File Details

### 🔹 `src/train.py`

This is the **main script** for running and logging an ML experiment using MLflow.
It performs the following steps:

* Loads dataset from `data/raw/`
* Reads parameters from `params.yaml`
* Trains a simple scikit-learn model
* Logs:

  * Parameters (learning rate, test size, etc.)
  * Metrics (accuracy, RMSE, etc.)
  * The trained model as an artifact

Example snippet (conceptual):

```python
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yaml

# Load params
params = yaml.safe_load(open("params.yaml"))
test_size = params["test_size"]

# Load data
df = pd.read_csv("data/raw/sample.csv")
X = df[["feature1", "feature2"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# Start MLflow run
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    
    mlflow.log_param("test_size", test_size)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")
```

---

### 🔹 `params.yaml`

Stores configurable parameters for experiments.
Example:

```yaml
test_size: 0.2
random_state: 42
```

---

### 🔹 `mlflow_inline_demo.ipynb`

An interactive notebook showing MLflow experiment logging inline.
Use this to:

* Understand MLflow APIs step-by-step
* Visualize metrics
* Inspect artifacts directly from notebook runs

Run it using:

```bash
jupyter notebook mlflow_inline_demo.ipynb
```

---

## 🧭 Step-by-Step Execution Flow (follow in sequence)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/varunchach/mlops_mlflow.git
cd mlops_mlflow
```

---

### 2️⃣ Create & Activate Virtual Environment

```bash
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

---

### 3️⃣ Start MLflow Server

Run the MLflow tracking server in a **separate terminal window**:

```bash
mlflow server `
    --backend-store-uri sqlite:///mlflow.db `
    --default-artifact-root file:./MLOps_Demo/mlruns `
    --host 127.0.0.1 `
    --port 5000
```

This will:

* Create a local MLflow database (`mlflow.db`)
* Store all experiment runs under `MLOps_Demo/mlruns`
* Start the MLflow UI at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

Keep this window open while executing your experiments.

---

### 4️⃣ Run the Training Script

Now in a new terminal (with your environment activated):

```bash
python .\src\train.py
```

This will:

* Train a simple ML model
* Log metrics, parameters, and the model to MLflow
* Create run entries visible in the MLflow UI

---

### 5️⃣ Explore MLflow UI

Open your browser and go to:
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

You’ll see:

* Logged parameters & metrics
* Saved model under **Artifacts**
* Each run’s unique ID and timestamp

You can compare runs and track model performance changes.

---

### 6️⃣ Modify and Re-Run

Change parameters in `params.yaml` (e.g., test size, random seed, algorithm type) and re-run:

```bash
python .\src\train.py
```

Each execution will log a **new run** in MLflow — allowing easy side-by-side comparison.

---

## ✅ Key Learnings

By the end of this demo, you will:

* Understand how to **track experiments** and **log metrics** with MLflow
* Know how to **store and retrieve models** as artifacts
* Be able to **compare multiple runs** in MLflow UI
* Learn to **configure MLflow’s tracking server** locally

---

## 🧾 Summary

This repository provides an end-to-end minimal **MLflow + Python** setup.
You’ll train models, log metadata, and explore experiment tracking — all on your local system, without any external dependencies.


