"""Model evaluation utilities for NBA draft prediction."""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Utility class for model evaluation operations."""
    
    def __init__(self):
        pass
    
    def evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
        """Evaluate model performance with multiple metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                logger.warning("Could not calculate AUC-ROC score")
                metrics['auc_roc'] = 0.0
        
        return metrics
    
    def compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Compare multiple model results."""
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.round(4)
        
        # Add ranking for each metric
        for metric in comparison_df.columns:
            comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False)
        
        return comparison_df
    
    def get_best_model(self, results: Dict[str, Dict[str, float]], metric: str = 'f1_score') -> str:
        """Get the best model based on a specific metric."""
        if not results:
            raise ValueError("No results provided")
        
        best_score = -1
        best_model = None
        
        for model_name, metrics in results.items():
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = model_name
        
        if best_model is None:
            raise ValueError(f"Metric '{metric}' not found in results")
        
        return best_model
