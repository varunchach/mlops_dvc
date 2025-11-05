# src/utils.py
import yaml

def read_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
