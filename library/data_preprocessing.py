

# ================Preprocessing================

#Preprocessing the data
#Converts Categorical data into binary values using One Hot Encoding

import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder

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

# # Creating Tables for each Challenge and their strategies
# for name,model in classifiers.items():
    
#     table_imbalance[name] = eval_table(['No Change','SMOTE','Cost Sensitive Learning'])
#     table_missing_values[name] = eval_table(['No Change','Dropping Missing Data','Imputing Data'])
#     table_outliers[name] = eval_table(['No Change','Winsorizing','Dropping Outlier'])
