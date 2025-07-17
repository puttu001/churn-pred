import pandas as pd
import joblib
import os
import yaml
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from urllib.parse import urlparse
import mlflow

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/puttu001/churn-pred.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="puttu001"
os.environ['MLFLOW_TRACKING_PASSWORD']="88f7090c916f46ab23742c5ba8e218c9138e82e9"

def hyperparameter_tuning(X_train,y_train,param_grid):
    lr=LogisticRegression()
    grid_search = GridSearchCV(estimator=lr,param_grid=param_grid,cv=5,n_jobs=-1,verbose =2)
    grid_search.fit(X_train,y_train)
    return grid_search

params = yaml.safe_load(open("params.yaml"))["train"]

def train(data_path,model_path,random_state,max_iter,penalty,solver,class_weight):
    data = pd.read_csv(data_path)
    X=data.drop(columns=["churn","customer_id","date_of_registration","pincode","state","city"])
    y=data['churn']
    X = pd.get_dummies(X)

    mlflow.set_tracking_uri("https://dagshub.com/puttu001/churn-pred.mlflow")

    with mlflow.start_run():
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
        signature = infer_signature(X_train,y_train)

        param_grid ={
            
          'penalty':['l1','l2'],
          'C': [0.01, 0.1, 1, 10, 100],
          'solver':['liblinear']
        }
        grid_search = hyperparameter_tuning(X_train,y_train,param_grid)

        best_model=grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        accuracy= accuracy_score(y_test,y_pred)
        print(f"Accuracy:{accuracy}")

        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_param("best C value",grid_search.best_params_['C'])
        mlflow.log_param("best penalty",grid_search.best_params_['penalty'])

        cm = confusion_matrix(y_test,y_pred)
        cr = classification_report(y_test,y_pred)

        mlflow.log_text(str(cm),"classification_matrix.txt")
        mlflow.log_text(str(cr),"classification_report.txt")


        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme



        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")




if __name__=='__main__':
    train(params['data'],
          params['model'],
          params['random_state'],
          params['max_iter'],
          params['penalty'],
          params['solver'],
          params['class_weight'])




