"""Tests for feature_engineer module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.preprocessing import StandardScaler
from nba_draft_utils.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engineer = FeatureEngineer()

    def test_init(self):
        """Test FeatureEngineer initialization."""
        assert isinstance(self.engineer.scaler, StandardScaler)
        assert self.engineer.fitted is False

    def test_create_basketball_features_basic(self):
        """Test creation of basic basketball features."""
        df = pd.DataFrame({
            'pts': [20, 15, 30],
            'fga': [15, 12, 25],
            'g': [10, 8, 15],
            'trb': [8, 6, 12],
            'ast': [5, 4, 7],
            'stl': [2, 1, 3],
            'blk': [1, 0, 2],
            'tov': [3, 2, 4]
        })
        
        result = self.engineer.create_basketball_features(df)
        
        # Check that original columns are preserved
        for col in df.columns:
            assert col in result.columns
        
        # Check points per FGA
        expected_ppfga = df['pts'] / (df['fga'] + 1)
        pd.testing.assert_series_equal(result['points_per_fga'], expected_ppfga, check_names=False)
        
        # Check per-game stats
        stats = ['pts', 'trb', 'ast', 'stl', 'blk']
        for stat in stats:
            expected_per_game = df[stat] / (df['g'] + 1)
            pd.testing.assert_series_equal(
                result[f'{stat}_per_game'], 
                expected_per_game, 
                check_names=False
            )
        
        # Check assist to turnover ratio
        expected_ast_tov = df['ast'] / (df['tov'] + 1)
        pd.testing.assert_series_equal(result['ast_tov_ratio'], expected_ast_tov, check_names=False)

    def test_create_basketball_features_missing_columns(self):
        """Test feature creation with missing columns."""
        df = pd.DataFrame({
            'pts': [20, 15, 30],
            'other_stat': [1, 2, 3]
        })
        
        result = self.engineer.create_basketball_features(df)
        
        # Should not create features that require missing columns
        assert 'points_per_fga' not in result.columns
        assert 'pts_per_game' not in result.columns
        assert 'ast_tov_ratio' not in result.columns
        
        # Original columns should be preserved
        for col in df.columns:
            assert col in result.columns

    def test_create_basketball_features_partial_columns(self):
        """Test feature creation with some required columns."""
        df = pd.DataFrame({
            'pts': [20, 15, 30],
            'fga': [15, 12, 25],
            'ast': [5, 4, 7],
            'tov': [3, 2, 4]
        })
        
        result = self.engineer.create_basketball_features(df)
        
        # Should create points_per_fga and ast_tov_ratio
        assert 'points_per_fga' in result.columns
        assert 'ast_tov_ratio' in result.columns
        
        # Should not create per-game stats (no 'g' column)
        assert 'pts_per_game' not in result.columns

    def test_create_basketball_features_zero_denominators(self):
        """Test feature creation handles zero denominators."""
        df = pd.DataFrame({
            'pts': [20, 15, 30],
            'fga': [0, 12, 25],  # Zero field goal attempts
            'g': [0, 8, 15],     # Zero games
            'ast': [5, 4, 7],
            'tov': [0, 2, 4]     # Zero turnovers
        })
        
        result = self.engineer.create_basketball_features(df)
        
        # Should handle division by adding 1 to denominators
        assert not result.isnull().any().any()
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()

    def test_create_basketball_features_empty_dataframe(self):
        """Test feature creation with empty DataFrame."""
        df = pd.DataFrame()
        result = self.engineer.create_basketball_features(df)
        assert result.empty

    def test_scale_features_train_only(self):
        """Test feature scaling with training data only."""
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        X_train_scaled, X_test_scaled = self.engineer.scale_features(X_train)
        
        # Check that scaler is fitted
        assert self.engineer.fitted is True
        
        # Check that training data is scaled
        assert X_train_scaled.shape == X_train.shape
        assert list(X_train_scaled.columns) == list(X_train.columns)
        
        # Check that means are approximately 0 and stds are approximately 1
        assert abs(X_train_scaled.mean().mean()) < 1e-10
        assert abs(X_train_scaled.std().mean() - 1.0) < 1e-10
        
        # X_test_scaled should be None
        assert X_test_scaled is None

    def test_scale_features_train_and_test(self):
        """Test feature scaling with both training and test data."""
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        X_test = pd.DataFrame({
            'feature1': [6, 7],
            'feature2': [60, 70]
        })
        
        X_train_scaled, X_test_scaled = self.engineer.scale_features(X_train, X_test)
        
        # Check that both datasets are scaled
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        
        # Check column names are preserved
        assert list(X_train_scaled.columns) == list(X_train.columns)
        assert list(X_test_scaled.columns) == list(X_test.columns)
        
        # Check that test data is transformed using training data statistics
        assert X_test_scaled is not None
        assert not X_test_scaled.isnull().any().any()

    def test_scale_features_preserves_index(self):
        """Test that feature scaling preserves DataFrame index."""
        custom_index = ['row1', 'row2', 'row3']
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [10, 20, 30]
        }, index=custom_index)
        
        X_train_scaled, _ = self.engineer.scale_features(X_train)
        
        # Check that index is preserved
        assert list(X_train_scaled.index) == custom_index

    def test_scale_features_single_column(self):
        """Test feature scaling with single column."""
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5]
        })
        
        X_train_scaled, _ = self.engineer.scale_features(X_train)
        
        assert X_train_scaled.shape == (5, 1)
        assert 'feature1' in X_train_scaled.columns

    def test_scale_features_identical_values(self):
        """Test feature scaling with identical values (zero variance)."""
        X_train = pd.DataFrame({
            'feature1': [5, 5, 5, 5, 5],
            'feature2': [1, 2, 3, 4, 5]
        })
        
        # This should not raise an error
        X_train_scaled, _ = self.engineer.scale_features(X_train)
        
        # Feature with zero variance should be scaled to 0
        assert all(X_train_scaled['feature1'] == 0)

    @patch('nba_draft_utils.feature_engineer.logger')
    def test_logging_feature_creation(self, mock_logger):
        """Test that feature creation logs info messages."""
        df = pd.DataFrame({
            'pts': [20, 15, 30],
            'fga': [15, 12, 25]
        })
        
        self.engineer.create_basketball_features(df)
        mock_logger.info.assert_called()

    def test_create_basketball_features_integration(self):
        """Test feature creation with realistic basketball data."""
        df = pd.DataFrame({
            'pts': [25.2, 18.7, 12.3],
            'fga': [18.5, 14.2, 9.8],
            'g': [82, 75, 65],
            'trb': [7.8, 5.2, 4.1],
            'ast': [8.5, 3.2, 2.1],
            'stl': [1.8, 1.1, 0.8],
            'blk': [0.9, 0.3, 2.1],
            'tov': [3.2, 2.1, 1.5]
        })
        
        result = self.engineer.create_basketball_features(df)
        
        # Check that we have the expected number of new features
        original_cols = len(df.columns)
        new_cols = len(result.columns)
        expected_new_features = 7  # points_per_fga + 5 per_game stats + ast_tov_ratio
        
        assert new_cols == original_cols + expected_new_features
        
        # Spot check some calculations
        assert result.loc[0, 'points_per_fga'] == pytest.approx(25.2 / 19.5, rel=1e-5)
        assert result.loc[0, 'pts_per_game'] == pytest.approx(25.2 / 83.0, rel=1e-5)
