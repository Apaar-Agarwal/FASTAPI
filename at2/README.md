# NBA Draft Utils

A Python package providing utilities for NBA draft prediction projects.

## Features

- **DataProcessor**: Data loading, cleaning, and preprocessing utilities
- **FeatureEngineer**: Basketball-specific feature creation and scaling
- **ModelEvaluator**: Model performance evaluation and comparison tools

## Installation

```bash
pip install nba-draft-utils
```

## Usage

```python
from nba_draft_utils import DataProcessor, FeatureEngineer, ModelEvaluator

# Load and clean data
processor = DataProcessor()
df = processor.load_data('data.csv')
df_clean = processor.clean_data(df)

# Engineer features
engineer = FeatureEngineer()
df_features = engineer.create_basketball_features(df_clean)

# Evaluate models
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(y_true, y_pred, y_prob)
```

## Development

This package is built with Poetry. To install for development:

```bash
poetry install
```

## License

MIT License
