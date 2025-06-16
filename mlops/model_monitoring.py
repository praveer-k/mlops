import numpy as np
from typing import Dict
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from mlops.feature_store import FeatureStore
from mlops.model_registry import ModelRegistry

class ModelMonitoring:
    def __init__(self, feature_store: FeatureStore, model_registry: ModelRegistry):
        self.feature_store = feature_store
        self.model_registry = model_registry
    
    def prediction_drift(self) -> float:
        """Test for Prediction Drift"""
        curr_data = self.feature_store.df[self.feature_store.feature_cols]
        X = self.feature_store.get_historical_features()
        if X is None:
            return 0
        hist_data = X[self.feature_store.feature_cols]
        if not hasattr(self.model_registry.model, "predict_proba"):
            raise Exception("Prediction probability cannot be created as model does not support it!")
        curr_preds = self.model_registry.model.predict_proba(curr_data)[:, 1]
        prev_preds = self.model_registry.model.predict_proba(hist_data)[:, 1]
        curr_preds = np.clip(curr_preds, 0, 1)
        prev_preds = np.clip(prev_preds, 0, 1)
        curr_hist, _ = np.histogram(curr_preds, bins=20, range=(0, 1), density=False)
        prev_hist, _ = np.histogram(prev_preds, bins=20, range=(0, 1), density=False)
        curr_prob_dist = curr_hist / (curr_hist.sum() + 1e-12)
        prev_prob_dist = prev_hist / (prev_hist.sum() + 1e-12)
        js_div = jensenshannon(curr_prob_dist + 1e-6, prev_prob_dist + 1e-6)
        print(f"Prediction Drift (JS Divergence): {js_div:.4f}")
        return round(js_div, 4)

    def concept_drift(self) -> Dict[str, Dict[str, float]]:
        """Test for Concept Drift"""
        X = self.feature_store.get_historical_features()
        X_curr = self.feature_store.df[self.feature_store.feature_cols]
        if X is None:
            return {feature: { "ks_stat":0, "p_value":0 } for feature in self.feature_store.feature_cols}
        cdrift = dict()
        for i, feature in enumerate(self.feature_store.feature_cols):
            ks_stat, p_value = ks_2samp(X_curr[feature], X[feature])
            cdrift[feature] = {
                "ks_stat": ks_stat,
                "p_value": p_value
            }
            print(f"Concept Drift (KS Test) for feature {feature}: {ks_stat:.4f}, P-Value: {p_value:.4f} (Drift if p < 0.05)")
        return cdrift
