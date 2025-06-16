
from airflow import DAG
from airflow.operators.python import PythonOperator

from mlops.data_validate import DataValidator
from mlops.feature_store import FeatureStore
from mlops.model_registry import ModelRegistry

class Orchestrator:
    def __init__(self):
        self.data_validator = DataValidator()
        self.feature_store = FeatureStore()
        self.model_registry = ModelRegistry()
        # self.model_trainer = ModelTrainer()
    
    def create_retraining_dag(self, model_name: str, schedule: str):
        """Create Airflow DAG for automated retraining"""
        with DAG(f'{model_name}_retraining', schedule_interval=schedule) as dag:
            # Data validation task
            validate_data = PythonOperator(
                task_id='validate_data',
                python_callable=self.data_validator.validate
            )
            
            # Feature engineering task  
            engineer_features = PythonOperator(
                task_id='engineer_features',
                python_callable=self.feature_store.transform
            )
            
            # Model training task
            train_model = PythonOperator(
                task_id='train_model',
                python_callable=self.model_trainer.train
            )
            
            # Model evaluation task
            evaluate_model = PythonOperator(
                task_id='evaluate_model',
                python_callable=self.model_registry.evaluate
            )
            
            # Model promotion task
            promote_model = PythonOperator(
                task_id='promote_model',
                python_callable=self.model_registry.log_to_mlflow
            )
            
            validate_data >> engineer_features >> train_model >> evaluate_model >> promote_model