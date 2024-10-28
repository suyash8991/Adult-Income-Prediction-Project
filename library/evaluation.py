import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           accuracy_score, confusion_matrix, classification_report)

def eval_table(row_names):
    """Create evaluation table with specified row names and metric columns."""
    return pd.DataFrame(
        index=row_names,
        columns=['Precision', 'Recall', 'F1 Score', 'Accuracy', 'Confusion Matrix']
    )

def initialize_evaluation_tables(classifiers):
    """Initialize evaluation tables for each classifier and challenge."""
    tables = {
        'imbalance': {},
        'missing_values': {},
        'outliers': {}
    }
    
    for name,models in classifiers.items():
        tables['imbalance'][name] = eval_table(['No Change', 'SMOTE', 'Cost Sensitive Learning'])
        tables['missing_values'][name] = eval_table(['No Change', 'Dropping Missing Data', 'Imputing Data'])
        tables['outliers'][name] = eval_table(['No Change', 'Winsorizing', 'Dropping Outlier'])
    
    return tables

def table_update(model, strategy, metrics_table, table_to_update):
    """Update evaluation table with metrics for given model and strategy."""
    for metric, value in metrics_table.items():
        table_to_update[model].at[strategy, metric] = value

def evaluate(classifiers, X_train, X_val, y_train, y_val, challenge, strategy, table_to_update):
    """Evaluate models with hyperparameter tuning and compute metrics."""
    results = {}
    
    for name, model_info in classifiers.items():
        model = model_info['model']
        params = model_info['params']

        # Hyperparameter tuning
        if params:
            random_search = RandomizedSearchCV(
                model, params, n_iter=10, cv=3, 
                scoring='accuracy', verbose=1, n_jobs=-1
            )
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
            print(f"Best hyperparameters for {name}: {random_search.best_params_}")
        else:
            best_model = model.fit(X_train, y_train)
        
        # Make predictions and calculate metrics
        y_pred = best_model.predict(X_val)
        
        results[name] = {
            'Precision': round(precision_score(y_val, y_pred) * 100, 2),
            'Recall': round(recall_score(y_val, y_pred) * 100, 2),
            'F1 Score': round(f1_score(y_val, y_pred) * 100, 2),
            'Accuracy': round(accuracy_score(y_val, y_pred) * 100, 2),
            'Confusion Matrix': confusion_matrix(y_val, y_pred)
        }
        
        print(f"Model {name}")
        print(classification_report(y_val, y_pred))
        
        # Update results
        table_update(name, strategy, results[name], table_to_update)
    
    return results