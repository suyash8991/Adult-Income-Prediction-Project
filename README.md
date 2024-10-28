# Adult Income Prediction Project

## Overview
This project analyzes the Adult Income Dataset to predict whether an individual's income exceeds $50K per year based on census data. The analysis focuses on addressing three key machine learning challenges: class imbalance, missing values, and outliers.

## Dataset Information
This project uses the Adult Income Dataset from the UCI Machine Learning Repository.

### Dataset Source
- **Name**: Adult Income Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)
- **Original Name**: Census Income Dataset
- **Year**: 1994
- **Donor**: Ronny Kohavi and Barry Becker, Data Mining and Visualization, Silicon Graphics

### Dataset Description
The dataset was extracted from the 1994 Census bureau database. The prediction task is to determine whether a person makes over $50K a year based on census data.
The dataset contains census data with the following features:
- Demographic attributes (age, gender, race, native country)
- Educational information (education level, educational-num)
- Employment details (workclass, occupation, hours-per-week)
- Financial indicators (capital-gain, capital-loss)
- Target variable: income >50K (binary classification)

Total records: 43,957




## Project Structure
```
adult_income_data_project/
├── config/
│   └── __init__.py
├── constant/
│   └── constants.py
├── data/
│   └── train.csv
├── library/
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── models.py
│   └── visualization.py
├── src/
│   ├── challenge_1.py
│   ├── challenge_2.py
│   ├── challenge_3.py
│   └── main.py
├── dataset_profile.py
├── README.md
└── requirements.txt
```

## Challenges Addressed

### 1. Class Imbalance
Three strategies implemented:
- Baseline (no changes)
- Over-sampling using SMOTE
- Cost-sensitive learning using class weights

### 2. Missing Values
Addressing missing data in workclass, occupation, and native-country fields:
- Baseline (no changes)
- Dropping rows with missing values
- Imputation using most frequent values

### 3. Outliers
Handling outliers in numerical features:
- Baseline (no changes)
- Winsorizing at specific percentiles:
  - capital-gain: 97th percentile
  - capital-loss: 97th percentile
  - hours-per-week: 95th percentile
  - age: 95th percentile
- Dropping identified outliers

## Key Findings
- Missing data analysis revealed:
  - Workclass: MCAR (Missing Completely at Random)
  - Occupation: NMAR (Not Missing at Random)
  - Native-country: MCAR
- Significant class imbalance with majority of records having income ≤50K
- Non-normal distribution in numerical features, particularly in capital-gain and capital-loss

## Usage
1. Clone the repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the dataset profile analysis:
```bash
python dataset_profile.py
```
4. Execute each challenge:
```bash
python -m challenge_1.py  # Run class imbalance challenge
python -m challenge_2.py  # Run missing values challenge
python -m challenge_3.py  # Run outliers challenge
```

