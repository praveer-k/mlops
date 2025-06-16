import os
import mlflow

from typing import Dict
from datetime import datetime

import pandas as pd
from sklearn.base import BaseEstimator
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

load_dotenv()

class ModelRegistry:
    def __init__(self, experiment_name: str, model_name: str, model: BaseEstimator, params: dict):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.model: BaseEstimator = model
        self.params = params
        self.client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"], registry_uri=os.environ["MLFLOW_S3_ENDPOINT_URL"])

    def get_model(self, model_name: str):
        latest_versions = self.client.get_latest_versions(name=model_name)
        if not latest_versions:
            raise ValueError(f"No model found with the name '{model_name}'")
        latest_version = max(latest_versions, key=lambda v: int(v.version))
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        metadata = {
            "name": latest_version.name,
            "version": latest_version.version,
            "run_id": latest_version.run_id,
            "stage": latest_version.current_stage,
            "creation_time": latest_version.creation_timestamp,
        }
        return model, metadata
    
    def evaluate(self, X_test_scaled, y_test) -> Dict[str, float]:
        y_pred = self.model.predict(X_test_scaled)
        y_scores = self.model.predict_proba(X_test_scaled)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_test, y_scores)
        }
        return metrics
    
    @staticmethod
    def is_model_ready_to_register(model_metrics: Dict[str, float], quality_metrics: Dict[str, float], data_drift: Dict[str, float], prediction_drift: float, concept_drift: Dict[str, Dict[str, float]]):
        """Is model ready to register"""
        concept_drift_is_stable = all(stats["p_value"] > 0.05 and stats["ks_stat"] < 0.1 for _, stats in concept_drift.items())
        data_drift_is_acceptable = all(stats < 0.3 for _, stats in data_drift.items())
        if (quality_metrics["missing_values"] / quality_metrics["sample_count"] > 0.05 or
            quality_metrics['duplicate_rows'] / quality_metrics["sample_count"] > 0.02):
            print("Warning! quality of the data is not good enough")

        ready_to_register = (model_metrics["auc_roc"] > 0.8 and 
            model_metrics["precision"] > 0.7 and 
            model_metrics["recall"] > 0.7 and 
            model_metrics["f1_score"] > 0.65 and 
            data_drift_is_acceptable and 
            prediction_drift < 0.3 and 
            concept_drift_is_stable 
        )
        return ready_to_register
    
    def get_all_metrics(self, model_metrics, quality_metrics, data_drift: Dict[str, float], prediction_drift, concept_drift: Dict[str, Dict[str, float]]):
        all_metrics = model_metrics | quality_metrics
        for k, v in data_drift.items():
            all_metrics[f"data_drift_{k}"] = v
        all_metrics["prediction_drift"] = prediction_drift
        for k, v in concept_drift.items():
                all_metrics[f"concept_drift_{k}_ks_stat"] = v["ks_stat"]
                all_metrics[f"concept_drift_{k}_p_value"] = v["p_value"]
        return all_metrics
    
    def log_to_mlflow(self, examples: pd.DataFrame, target_col, feature_cols, model_metrics, quality_metrics, data_drift: Dict[str, float], prediction_drift, concept_drift: Dict[str, Dict[str, float]]):
        """Log model to Model Registry"""
        metrics = model_metrics, quality_metrics, data_drift, prediction_drift, concept_drift
        
        # if not self.is_model_ready_to_register(*metrics):
        #     print("Metrics indicate that the model is not ready for deployment")
        #     return
        
        all_metrics = self.get_all_metrics(*metrics)

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run():
            mlflow.log_params(self.params)
            mlflow.log_metrics(all_metrics)
            mlflow.sklearn.log_model(
                sk_model=self.model, 
                name=self.model_name,
                input_example=examples,
                registered_model_name=self.model_name
            )
            mlflow.set_tags({
                "framework": "scikit-learn",
                "model_type": self.model.__class__.__name__,
                "experiment": self.experiment_name,
                "features_used": ",".join(feature_cols),
                "target_column": target_col,
                "timestamp": str(datetime.now())
            })
            print("Logged to MLflow")
