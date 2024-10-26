# #### Suyash Sreekumar
# #### Dataset : Adult Income Dataset
# #### Final Project
# #### Date : 03/18/24 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score,accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, cross_validate, RandomizedSearchCV
from imblearn.over_sampling import SMOTENC
from scipy.stats.mstats import winsorize
from main import random_state
'''Functions present here 
1. datapreprocessing
2. standardize
3. eval_table 
4. table_update
5. evaluate
'''
np.random.seed(random_state)


#  ================ Model Declaration ================

# Classifier dictionary for models being used
classifiers = {
    "Dummy Classifier (Most frequent)": {
        'model': DummyClassifier(strategy='most_frequent', random_state=random_state),
        'params': {}
    },
    "Dummy Classifier (Stratified)": {
        'model': DummyClassifier(strategy='stratified', random_state=random_state),
        'params': {}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=random_state,n_jobs=-1),
        'params': {
            'n_estimators': [50, 100, 200],
            'min_samples_split': [2, 5, 10],
            'max_features':[.2,.5,.7]
        }
    }
}
#Creating dict for each table which will be used to store metrics table for each model
table_imbalance={
    'Dummy Classifier (Most frequent)': '',
    'Dummy Classifier (Stratified)' : '',
    'Random Forest': ''}

table_missing_values={
    'Dummy Classifier (Most frequent)': '',
    'Dummy Classifier (Stratified)' : '',
    'Random Forest': ''
}
table_outliers={
    'Dummy Classifier (Most frequent)': '',
    'Dummy Classifier (Stratified)' : '',
    'Random Forest': ''
}



# ================Preprocessing================

#Preprocessing the data
#Converts Categorical data into binary values using One Hot Encoding

def datapreprocessing(data,data_test):
    #Education num and Education are similar and hence dropping cateogorical equivalent of education-num
    data.drop(['education'],errors='ignore',axis=1,inplace=True)
    data_test.drop(['education'],errors='ignore',axis=1,inplace=True)
    print("X ",len(data))
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Apply One-Hot Encoding to categorical columns; drop first done to reduce one column from each category
    # OHE does not ignore missing values but creates additional label for it as nan and tags it.
    one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    # Transform the categorical columns and replace them in the original dataframe
    data_encoded = pd.DataFrame(one_hot_encoder.fit_transform(data[categorical_columns]), columns=one_hot_encoder.get_feature_names_out(categorical_columns))

    print("Shape before one-hot encoding:", data.shape)
    data.reset_index(drop=True, inplace=True)
    data = pd.concat([data, data_encoded], axis=1)
    data.drop(categorical_columns, axis=1, inplace=True)

    print("Shape after one-hot encoding:", data.shape)    


    data_test_encoded = pd.DataFrame(one_hot_encoder.transform(data_test[categorical_columns]),columns=one_hot_encoder.get_feature_names_out(categorical_columns))
    print("Shape before one-hot encoding test:", data_test.shape)    
    data_test.reset_index(drop=True, inplace=True)

    data_test = pd.concat([data_test, data_test_encoded], axis=1)
    data_test.drop(categorical_columns, axis=1, inplace=True)
    print("Shape after one-hot encoding test :", data_test.shape)    

    return data,data_test
    

# ================ Standardize function ================

#takes X,Xval data and standardizes it

def standardize(X1,X2):
    X=X1.copy()
    X_val=X2.copy()
    # Getting list of numerical columns 
    int_col = X1.select_dtypes(include=['int']).columns

    X_columns_for_std = int_col
    
    scaler = StandardScaler()
    std = scaler.fit(X[X_columns_for_std])
    X[X_columns_for_std] = std.transform(X[X_columns_for_std])
    X_val[X_columns_for_std] = std.transform(X_val[X_columns_for_std])

    
    return X,X_val

# Function that takes row names (Each Challenge strategies and columns as Precision, 
# Recall,F1 Score,AUC-ROC, Accuracy)
def eval_table(row_names):
    
    results=pd.DataFrame(index=row_names,columns=['Precision','Recall','F1 Score','Accuracy','Confusion Matrix'])
    return results

# Creating Tables for each Challenge and their strategies
for name,model in classifiers.items():
    
    table_imbalance[name] = eval_table(['No Change','SMOTE','Cost Sensitive Learning'])
    table_missing_values[name] = eval_table(['No Change','Dropping Missing Data','Imputing Data'])
    table_outliers[name] = eval_table(['No Change','Winsorizing','Dropping Outlier'])
  



# ================ Function Table update ================
"""
    Update the evaluation table with metrics for a given model, challenge, and strategy.

    Parameters:
        model (str): The name of the model.
        challenge (str): The name of the challenge (e.g., 'Class Imbalance', 'Missing Values', etc.).
        strategy (str): The name of the strategy used (e.g., 'No Change', 'SMOTE', 'Cost Sensitive Learning', etc.).
        metrics_table (dict): A dictionary containing evaluation metrics for the model, challenge, and strategy.
        evaluation_method: The evaluation method used ; used to update corresponding table.['train_test_split','stratified_split']
        table_to_ipdate : either table_imbalance or table_missing_values or  table_outliers
    Returns:
        None
"""
def table_update(model,challenge,strategy,metrics_table,evaluation_method,table_to_update):
#     print(f"Table update called : {challenge} - {strategy}")
    for metric,value in metrics_table.items():
        table_to_update[model].at[strategy,metric]=value
    # if(challenge=='Class Imbalance'):
    #     for metric,value in metrics_table.items():
    #         table_imbalance[model].at[strategy,metric]=value

            
    # elif(challenge == "Missing Values"):
    #     for metric,value in metrics_table.items():
    #         table_missing_values[model].at[strategy,metric]=value    
       

    # else :
    #     for metric,value in metrics_table.items():
    #         table_outliers[model].at[strategy,metric]=value    

# ================ Function Evaluate ================

'''Evaluate function : to evaluate  different models with metrics precision, recall, f1 score, accuracy 
    and confusion matrix 
    Hyperparameters are tuned here using randomized search for each model
    Inputs taken : classifiers_dict, X_train, X_val, y_train, y_val, challenge, strategy'''
def evaluate(classifiers, X_train, X_val, y_train, y_val, challenge, strategy,table_to_update):
    results = {}
    stratified_results={}
    for name, model_info in classifiers.items():
        model = model_info['model']
        params = model_info['params']

        # Hyperparameter tuning using GridSearchCV
        if params: 
            random_search = RandomizedSearchCV(model, params, n_iter=10, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
                
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
            print(f"Best hyperparameters for {name}: {random_search.best_params_}")
        # if no parameters specified, just fit the data
        else: 
                
            best_model = model.fit(X_train, y_train)
     
        # Predict on validation set
        y_pred = best_model.predict(X_val)

        # Calculate evaluation metrics
        precision = round(precision_score(y_val, y_pred) * 100, 2)
        recall = round(recall_score(y_val, y_pred) * 100, 2)
        f1 = round(f1_score(y_val, y_pred) * 100, 2)
        accuracy = round(accuracy_score(y_val, y_pred) * 100, 2)
        confusion = confusion_matrix(y_val, y_pred)
        print("Model ",name)
        print(classification_report(y_val, y_pred))


        # Store results
        results[name] = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Accuracy': accuracy,
            'Confusion Matrix':confusion
        }
 

        # Update resutls of train test split
        table_update(name, challenge, strategy, results[name],'train_test_split',table_to_update)

    return results


# ================ Function Plot Confusion Matrix ================

def plot_confusion_matrix(challenge,strategy):
    table=table_imbalance
    strategy_index = 0

    if(challenge=='Class Imbalance'):
        table=table_imbalance
    elif(challenge=='Missing Values'):
        table=table_missing_values
    else :
        table=table_outliers
    
    if(strategy in ['No Change']):
        strategy_index = 0
    elif(strategy in ['SMOTE', 'Dropping Missing Data','Winsorizing']):
        strategy_index = 1
    else :
        strategy_index = 2
    print(f"\nChallenge {challenge} - Strategy {strategy} :Confusion Matrix for Random forest \n")

    plt.figure(figsize=(6, 4))
    
    # showing confusion matrix using heatmap
    sns.heatmap(table['Random Forest']['Confusion Matrix'][strategy_index], annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title('Confusion Matrix')
    plt.show()


# For cost sensitive learning
# Classifier weights dict for applying class weights in Random Forest 
    
classifiers_weights = {
    "Dummy Classifier (Most frequent)": {
        'model': DummyClassifier(strategy='most_frequent', random_state=random_state),
        'params': {}
    },
    "Dummy Classifier (Stratified)": {
        'model': DummyClassifier(strategy='stratified', random_state=random_state),
        'params': {}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=random_state,class_weight={1:3,0:1},n_jobs=-1),

        'params': {
            'n_estimators': [50, 100, 200],
            'min_samples_split': [2, 5, 10],
            'max_features':[.2,.5,.7]
        }
    }
}