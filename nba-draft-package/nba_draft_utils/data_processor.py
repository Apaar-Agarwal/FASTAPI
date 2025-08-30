"""Data processing utilities for NBA draft prediction."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Utility class for data processing operations."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by handling missing values and duplicates."""
        df_clean = df.copy()
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # Remove duplicates
        df_clean.drop_duplicates(inplace=True)
        
        logger.info(f"Cleaned data shape: {df_clean.shape}")
        return df_clean
    
    def split_features_target(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Split DataFrame into features and target."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        return X, y
