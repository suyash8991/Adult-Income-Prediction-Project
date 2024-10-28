from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from config import RANDOM_STATE

# Base classifier configurations without class weights
BASE_CLASSIFIER_CONFIG = {
    "Dummy Classifier (Most frequent)": {
        'model': DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE),
        'params': {}
    },
    "Dummy Classifier (Stratified)": {
        'model': DummyClassifier(strategy='stratified', random_state=RANDOM_STATE),
        'params': {}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'params': {
            'n_estimators': [50, 100, 200],
            'min_samples_split': [2, 5, 10],
            'max_features': [.2, .5, .7]
        }
    }
}

# Classifier configurations with class weights for imbalanced data
WEIGHTED_CLASSIFIER_CONFIG = {
    "Dummy Classifier (Most frequent)": {
        'model': DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE),
        'params': {}
    },
    "Dummy Classifier (Stratified)": {
        'model': DummyClassifier(strategy='stratified', random_state=RANDOM_STATE),
        'params': {}
    },
    'Random Forest': {
        'model': RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight={1: 3, 0: 1},
            n_jobs=-1
        ),
        'params': {
            'n_estimators': [50, 100, 200],
            'min_samples_split': [2, 5, 10],
            'max_features': [.2, .5, .7]
        }
    }
}