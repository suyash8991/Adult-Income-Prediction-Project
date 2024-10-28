# Configuration settings
RANDOM_STATE = 42
DATA_PATH = '../data/train.csv'

# Model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'min_samples_split': [2, 5, 10],
        'max_features': [0.2, 0.5, 0.7]
    }
}