from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

def get_base_classifiers(random_state):
    """Return dictionary of base classifiers with their parameters."""
    return {
        "Dummy Classifier (Most frequent)": {
            'model': DummyClassifier(strategy='most_frequent', random_state=random_state),
            'params': {}
        },
        "Dummy Classifier (Stratified)": {
            'model': DummyClassifier(strategy='stratified', random_state=random_state),
            'params': {}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=random_state, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100, 200],
                'min_samples_split': [2, 5, 10],
                'max_features': [.2, .5, .7]
            }
        }
    }

def get_weighted_classifiers(random_state):
    """Return dictionary of classifiers with class weights for imbalanced data."""
    return {
        "Dummy Classifier (Most frequent)": {
            'model': DummyClassifier(strategy='most_frequent', random_state=random_state),
            'params': {}
        },
        "Dummy Classifier (Stratified)": {
            'model': DummyClassifier(strategy='stratified', random_state=random_state),
            'params': {}
        },
        'Random Forest': {
            'model': RandomForestClassifier(
                random_state=random_state,
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