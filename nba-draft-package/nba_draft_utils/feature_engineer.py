"""Feature engineering utilities for NBA draft prediction."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Utility class for feature engineering operations."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def create_basketball_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basketball-specific features."""
        df_features = df.copy()
        
        # Shooting efficiency
        if all(col in df.columns for col in ['pts', 'fga']):
            df_features['points_per_fga'] = df['pts'] / (df['fga'] + 1)
        
        # Per-game stats
        if 'g' in df.columns:
            stats = ['pts', 'trb', 'ast', 'stl', 'blk']
            for stat in stats:
                if stat in df.columns:
                    df_features[f'{stat}_per_game'] = df[stat] / (df['g'] + 1)
        
        # Ratios
        if all(col in df.columns for col in ['ast', 'tov']):
            df_features['ast_tov_ratio'] = df['ast'] / (df['tov'] + 1)
        
        logger.info(f"Created {len(df_features.columns) - len(df.columns)} new features")
        return df_features
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features using StandardScaler."""
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        self.fitted = True
        return X_train_scaled, X_test_scaled
