import duckdb
import functools
import numpy as np
import pandas as pd

from typing import Dict
from datetime import datetime
from feast import FeatureStore as FeastFeatureStore, FeatureView
from scipy.stats import ks_2samp

class FeatureStore:
    def __init__(self, table_name: str, df: pd.DataFrame, target_col: str, feature_cols: list[str], entity_cols: list[str], db_path: str = f"local/offline_datastore.duckdb", fs_path: str = "feature_store.yaml"):
        self.table_name = table_name
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.entity_cols = entity_cols
        self.db_path = db_path
        self.fs_path = fs_path
        self.store = FeastFeatureStore(fs_yaml_file=self.fs_path)
        self.conn = duckdb.connect(database=self.db_path, read_only=False)
        self.last_creation_timestamp = self.get_last_creation_timestamp()

    def _check_if_table_exists(self):
        table_exists = self.conn.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{self.table_name}'").fetchone()[0] > 0
        return table_exists
    
    def get_last_creation_timestamp(self):
        last_timestamp = None
        if self._check_if_table_exists():
            result = self.conn.execute(f"SELECT MAX(creation_timestamp) FROM {self.table_name}").fetchone()
            last_timestamp = result[0]
            return last_timestamp
        return None

    def append_offline_features(self):
        df_with_timestamp = self.df.copy()
        df_with_timestamp['creation_timestamp'] = pd.to_datetime(datetime.now())
        if not self._check_if_table_exists():
            self.conn.execute(f"CREATE TABLE {self.table_name} AS SELECT * FROM df_with_timestamp LIMIT 0")
        self.conn.append(self.table_name, df_with_timestamp)
        print(f"Data added to offline datastore in table '{self.table_name}'.")

    def push_online_features(self, feature_view_name: str, data: pd.DataFrame):
        if "event_timestamp" not in data.columns:
            data["event_timestamp"] = datetime.now()
        self.store.push(feature_view_name, data)
        print(f"Pushed {len(data)} rows to '{feature_view_name}'")

    def get_historical_features(self):
        if self._check_if_table_exists():
            historical_df = self.conn.execute(f"""SELECT * FROM {self.table_name} WHERE creation_timestamp = ?""", [self.last_creation_timestamp]).fetchdf()
            return historical_df[self.feature_cols]
        return None
    
    def quality_metrics(self) -> Dict[str, int]:
        """Get Data Quality Metrics"""
        X = self.df[self.feature_cols + self.entity_cols]
        quality_metrics = {
            'missing_values': X.isnull().sum().sum(),
            'duplicate_rows': X.duplicated().sum(),
            'feature_count': len(self.feature_cols),
            'sample_count': len(X)
        }
        print(f"Data quality metrics: {quality_metrics}")
        return quality_metrics
    
    def data_drift(self) -> Dict[str, Dict[str, float]]:
        """Test for Data Drift"""
        X = self.get_historical_features()
        X_curr = self.df[self.feature_cols]
        if X is None:
            return {feature: { "ks_stat":0, "p_value":0 } for feature in self.feature_cols}
        cdrift = dict()
        for i, feature in enumerate(self.feature_cols):
            ks_stat, p_value = ks_2samp(X_curr[feature], X[feature])
            cdrift[feature] = {
                "ks_stat": ks_stat,
                "p_value": p_value
            }
            print(f"Data Drift (KS Test) for feature {feature}: {ks_stat:.4f}, P-Value: {p_value:.4f} (Drift if p < 0.05)")
        return cdrift
       
    @staticmethod
    def is_data_quality_acceptable(quality_metrics, data_drift: Dict[str, Dict[str, float]]) -> bool:
        """Is data quality acceptable"""
        data_drift_is_acceptable = all(stats["p_value"] > 0.05 and stats["ks_stat"] < 0.1 for _, stats in data_drift.items())
        if (quality_metrics["missing_values"] / quality_metrics["sample_count"] <= 0.05 and
            quality_metrics['duplicate_rows'] / quality_metrics["sample_count"] <= 0.02 and
            data_drift_is_acceptable
        ):
            print("Quality of the data is acceptable")
            return True
        print("Warning! Data Quality if not good enough to push to feature store")
        return False
    
    @staticmethod
    def transform(func, feature_view: FeatureView):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            print(f"Name of the feature view: {feature_view.name}")
            # Add code here to extend the capabilites to store features
            # step 1. check quality and drift
            # step 2. throw error if the data quality is not upto the mark
            # step 3. if data quality is acceptable then store data to offline store
            # step 4. optionally push to online store given model quality is acceptable
            # step 5. pickle and save transformation method somewhere on the object store in a versioned bucket            
            return result
        return wrapper