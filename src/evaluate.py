import pandas as pd
import os
import yaml
import joblib
from sklearn.metrics import accuracy_score
import mlflow
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/puttu001/churn-pred.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="puttu001"
os.environ['MLFLOW_TRACKING_PASSWORD']="88f7090c916f46ab23742c5ba8e218c9138e82e9"

params = yaml.safe_load(open("params.yaml"))["train"]
def evaluate(data_path,model_path):
    data = pd.read_csv(data_path)
    X=data.drop(columns=["churn","customer_id","date_of_registration","pincode","state","city"])
    y=data['churn']
    X = pd.get_dummies(X)

    mlflow.set_tracking_uri("https://dagshub.com/puttu001/churn-pred.mlflow")
    model = joblib.load(model_path)

    predictions = model.predict(X)
    accuracy = accuracy_score(y,predictions)

    mlflow.log_metric("accuracy",accuracy)
    print(f"model accuracy:{accuracy}")

if __name__=="__main__":
    evaluate(params["data"],params["model"])


