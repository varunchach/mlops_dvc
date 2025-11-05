# create_data.py
from sklearn.datasets import load_iris
import pandas as pd, os

def main():
    data = load_iris()
    df = pd.DataFrame(data=data["data"],
                      columns=["sepal_length","sepal_width","petal_length","petal_width"])
    df["target"] = data["target"]
    os.makedirs("C:/Users/Satej Raste/Downloads/MLOps_Demo/data/raw", exist_ok=True)
    path = "C:/Users/Satej Raste/Downloads/MLOps_Demo/data/raw/iris.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {path} with {len(df)} rows")

if __name__ == "__main__":
    main()
