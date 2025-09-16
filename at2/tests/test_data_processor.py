"""Tests for data_processor module."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch
from nba_draft_utils.data_processor import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DataProcessor(random_state=42)

    def test_init(self):
        """Test DataProcessor initialization."""
        assert self.processor.random_state == 42
        
        # Test with different random state
        processor2 = DataProcessor(random_state=123)
        assert processor2.random_state == 123

    def test_load_data_valid_csv(self):
        """Test loading valid CSV data."""
        # Create a temporary CSV file
        test_data = pd.DataFrame({
            'name': ['Player1', 'Player2', 'Player3'],
            'points': [20, 15, 30],
            'assists': [5, 8, 3]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            result = self.processor.load_data(temp_file)
            pd.testing.assert_frame_equal(result, test_data)
        finally:
            os.unlink(temp_file)

    def test_load_data_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.processor.load_data('non_existent_file.csv')

    def test_clean_data_with_missing_values(self):
        """Test data cleaning with missing values."""
        df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0],
            'categorical_col': ['A', 'B', None, 'A', 'B'],
            'target': [1, 0, 1, 0, 1]
        })
        
        result = self.processor.clean_data(df)
        
        # Check that missing values are filled
        assert not result.isnull().any().any()
        
        # Check that numeric column is filled with median
        expected_median = df['numeric_col'].median()  # 2.5
        assert result.loc[2, 'numeric_col'] == expected_median
        
        # Check that categorical column is filled with mode
        expected_mode = df['categorical_col'].mode()[0]  # 'A' or 'B'
        assert result.loc[2, 'categorical_col'] in ['A', 'B']

    def test_clean_data_with_duplicates(self):
        """Test data cleaning removes duplicates."""
        df = pd.DataFrame({
            'col1': [1, 2, 1, 3],
            'col2': ['A', 'B', 'A', 'C'],
            'target': [1, 0, 1, 0]
        })
        
        result = self.processor.clean_data(df)
        
        # Should remove one duplicate row
        assert len(result) == 3
        assert not result.duplicated().any()

    def test_clean_data_empty_dataframe(self):
        """Test cleaning empty DataFrame."""
        df = pd.DataFrame()
        result = self.processor.clean_data(df)
        assert result.empty

    def test_clean_data_no_mode_categorical(self):
        """Test cleaning categorical column with no mode."""
        df = pd.DataFrame({
            'categorical_col': [np.nan, np.nan, np.nan],
            'numeric_col': [1, 2, 3]
        })
        
        result = self.processor.clean_data(df)
        
        # Should fill with 'Unknown'
        assert all(result['categorical_col'] == 'Unknown')

    def test_split_features_target_valid(self):
        """Test splitting features and target."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        
        X, y = self.processor.split_features_target(df, 'target')
        
        expected_X = df[['feature1', 'feature2']]
        expected_y = df['target']
        
        pd.testing.assert_frame_equal(X, expected_X)
        pd.testing.assert_series_equal(y, expected_y)

    def test_split_features_target_invalid_column(self):
        """Test splitting with invalid target column."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        with pytest.raises(ValueError, match="Target column 'target' not found"):
            self.processor.split_features_target(df, 'target')

    def test_split_features_target_single_column(self):
        """Test splitting DataFrame with only target column."""
        df = pd.DataFrame({
            'target': [0, 1, 0]
        })
        
        X, y = self.processor.split_features_target(df, 'target')
        
        # X should be empty DataFrame
        assert X.empty
        assert len(X.columns) == 0
        pd.testing.assert_series_equal(y, df['target'])

    @patch('nba_draft_utils.data_processor.logger')
    def test_logging_info_messages(self, mock_logger):
        """Test that info messages are logged correctly."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'target': [0, 1, 0]
        })
        
        # Test load_data logging
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            self.processor.load_data(temp_file)
            mock_logger.info.assert_called()
        finally:
            os.unlink(temp_file)
        
        # Test clean_data logging
        self.processor.clean_data(df)
        mock_logger.info.assert_called()

    @patch('nba_draft_utils.data_processor.logger')
    def test_logging_error_messages(self, mock_logger):
        """Test that error messages are logged correctly."""
        with pytest.raises(FileNotFoundError):
            self.processor.load_data('non_existent_file.csv')
        
        mock_logger.error.assert_called()
