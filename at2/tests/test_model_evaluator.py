"""Tests for model_evaluator module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from nba_draft_utils.model_evaluator import ModelEvaluator


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator()

    def test_init(self):
        """Test ModelEvaluator initialization."""
        assert isinstance(self.evaluator, ModelEvaluator)

    def test_evaluate_model_basic_binary_classification(self):
        """Test basic model evaluation for binary classification."""
        y_true = pd.Series([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0])
        
        metrics = self.evaluator.evaluate_model(y_true, y_pred)
        
        # Check that all basic metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert 0 <= metrics[metric] <= 1
        
        # Manually verify accuracy
        expected_accuracy = 4/5  # 4 correct out of 5
        assert metrics['accuracy'] == expected_accuracy

    def test_evaluate_model_with_probabilities(self):
        """Test model evaluation with probability scores."""
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.8, 0.9, 0.2])
        
        metrics = self.evaluator.evaluate_model(y_true, y_pred, y_prob)
        
        # Check that AUC-ROC is included
        assert 'auc_roc' in metrics
        assert isinstance(metrics['auc_roc'], float)
        assert 0 <= metrics['auc_roc'] <= 1

    def test_evaluate_model_multiclass(self):
        """Test model evaluation for multiclass classification."""
        y_true = pd.Series([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])
        
        metrics = self.evaluator.evaluate_model(y_true, y_pred)
        
        # Check that all metrics are calculated (using weighted average)
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            assert metric in metrics
            assert isinstance(metrics[metric], float)

    def test_evaluate_model_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        y_true = pd.Series([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        
        metrics = self.evaluator.evaluate_model(y_true, y_pred)
        
        # All metrics should be 1.0 for perfect predictions
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0

    def test_evaluate_model_all_wrong_predictions(self):
        """Test evaluation with all wrong predictions."""
        y_true = pd.Series([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        
        metrics = self.evaluator.evaluate_model(y_true, y_pred)
        
        # Accuracy should be 0.0
        assert metrics['accuracy'] == 0.0

    def test_evaluate_model_edge_case_single_class_true(self):
        """Test evaluation when true labels have only one class."""
        y_true = pd.Series([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 1])
        
        metrics = self.evaluator.evaluate_model(y_true, y_pred)
        
        # Should handle gracefully without errors
        assert 'accuracy' in metrics
        assert metrics['accuracy'] == 0.75

    def test_evaluate_model_with_zero_division(self):
        """Test evaluation handles zero division cases."""
        y_true = pd.Series([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        
        metrics = self.evaluator.evaluate_model(y_true, y_pred)
        
        # Should handle zero division gracefully
        assert all(isinstance(v, float) for v in metrics.values())
        assert not any(np.isnan(v) for v in metrics.values())

    @patch('nba_draft_utils.model_evaluator.logger')
    def test_evaluate_model_auc_warning(self, mock_logger):
        """Test that AUC calculation warnings are logged."""
        y_true = pd.Series([0, 0, 0])  # Only one class
        y_pred = np.array([0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3])
        
        metrics = self.evaluator.evaluate_model(y_true, y_pred, y_prob)
        
        # Should log warning and set AUC to 0.0
        mock_logger.warning.assert_called_with("Could not calculate AUC-ROC score")
        assert metrics['auc_roc'] == 0.0

    def test_compare_models_basic(self):
        """Test basic model comparison."""
        results = {
            'model1': {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.85},
            'model2': {'accuracy': 0.85, 'precision': 0.8, 'recall': 0.8},
            'model3': {'accuracy': 0.75, 'precision': 0.7, 'recall': 0.9}
        }
        
        comparison = self.evaluator.compare_models(results)
        
        # Check basic structure
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3
        assert list(comparison.index) == ['model1', 'model2', 'model3']
        
        # Check that original metrics are preserved
        assert 'accuracy' in comparison.columns
        assert 'precision' in comparison.columns
        assert 'recall' in comparison.columns
        
        # Check that rankings are added
        assert 'accuracy_rank' in comparison.columns
        assert 'precision_rank' in comparison.columns
        assert 'recall_rank' in comparison.columns

    def test_compare_models_ranking(self):
        """Test that model ranking is correct."""
        results = {
            'model1': {'accuracy': 0.8},
            'model2': {'accuracy': 0.9},  # Best
            'model3': {'accuracy': 0.7}   # Worst
        }
        
        comparison = self.evaluator.compare_models(results)
        
        # Check rankings (higher score = better rank = lower rank number)
        assert comparison.loc['model2', 'accuracy_rank'] == 1  # Best
        assert comparison.loc['model1', 'accuracy_rank'] == 2  # Middle
        assert comparison.loc['model3', 'accuracy_rank'] == 3  # Worst

    def test_compare_models_empty_results(self):
        """Test model comparison with empty results."""
        results = {}
        comparison = self.evaluator.compare_models(results)
        
        assert comparison.empty

    def test_compare_models_rounding(self):
        """Test that results are properly rounded."""
        results = {
            'model1': {'accuracy': 0.123456789}
        }
        
        comparison = self.evaluator.compare_models(results)
        
        # Should be rounded to 4 decimal places
        assert comparison.loc['model1', 'accuracy'] == 0.1235

    def test_get_best_model_basic(self):
        """Test getting the best model by default metric (f1_score)."""
        results = {
            'model1': {'f1_score': 0.8},
            'model2': {'f1_score': 0.9},  # Best
            'model3': {'f1_score': 0.7}
        }
        
        best_model = self.evaluator.get_best_model(results)
        assert best_model == 'model2'

    def test_get_best_model_custom_metric(self):
        """Test getting the best model by custom metric."""
        results = {
            'model1': {'accuracy': 0.8, 'precision': 0.6},
            'model2': {'accuracy': 0.7, 'precision': 0.9},  # Best precision
            'model3': {'accuracy': 0.9, 'precision': 0.5}   # Best accuracy
        }
        
        # Test with accuracy
        best_accuracy = self.evaluator.get_best_model(results, 'accuracy')
        assert best_accuracy == 'model3'
        
        # Test with precision
        best_precision = self.evaluator.get_best_model(results, 'precision')
        assert best_precision == 'model2'

    def test_get_best_model_empty_results(self):
        """Test getting best model with empty results."""
        with pytest.raises(ValueError, match="No results provided"):
            self.evaluator.get_best_model({})

    def test_get_best_model_missing_metric(self):
        """Test getting best model with missing metric."""
        results = {
            'model1': {'accuracy': 0.8},
            'model2': {'accuracy': 0.9}
        }
        
        with pytest.raises(ValueError, match="Metric 'f1_score' not found in results"):
            self.evaluator.get_best_model(results, 'f1_score')

    def test_get_best_model_partial_metric_coverage(self):
        """Test getting best model when not all models have the metric."""
        results = {
            'model1': {'accuracy': 0.8, 'precision': 0.7},
            'model2': {'accuracy': 0.9},  # Missing precision
            'model3': {'precision': 0.95}  # Missing accuracy
        }
        
        # Should work with accuracy (models 1 and 2 have it)
        best_accuracy = self.evaluator.get_best_model(results, 'accuracy')
        assert best_accuracy == 'model2'
        
        # Should work with precision (models 1 and 3 have it)
        best_precision = self.evaluator.get_best_model(results, 'precision')
        assert best_precision == 'model3'

    def test_get_best_model_tie_handling(self):
        """Test that ties are handled consistently."""
        results = {
            'model1': {'accuracy': 0.8},
            'model2': {'accuracy': 0.8},
            'model3': {'accuracy': 0.7}
        }
        
        best_model = self.evaluator.get_best_model(results, 'accuracy')
        # Should return one of the tied models (first one encountered)
        assert best_model in ['model1', 'model2']

    def test_integration_full_workflow(self):
        """Test full workflow integration."""
        # Simulate predictions from multiple models
        y_true = pd.Series([0, 1, 1, 0, 1, 0, 1, 0])
        
        model_predictions = {
            'logistic_regression': np.array([0, 1, 1, 0, 0, 0, 1, 0]),
            'random_forest': np.array([0, 1, 1, 0, 1, 0, 1, 1]),
            'svm': np.array([0, 1, 0, 0, 1, 0, 1, 0])
        }
        
        # Evaluate all models
        results = {}
        for model_name, predictions in model_predictions.items():
            results[model_name] = self.evaluator.evaluate_model(y_true, predictions)
        
        # Compare models
        comparison = self.evaluator.compare_models(results)
        assert len(comparison) == 3
        
        # Get best model
        best_model = self.evaluator.get_best_model(results)
        assert best_model in model_predictions.keys()
