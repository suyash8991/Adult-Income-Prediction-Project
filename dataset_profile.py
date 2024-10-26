#!/usr/bin/env python
# coding: utf-8

# #### Suyash Sreekumar
# #### Dataset : Adult Income Dataset
# #### Final Project
# #### Date : 03/18/24 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('/home/suyash/Winter_24/ML_Challenges/Final_Project/income_dataset/train.csv')
df.head()


# In[3]:


df.info()


# ###  Total rows : 43957

# In[4]:


df.describe()


# #### Analysing Categorical features

# In[5]:


columns=df.columns
categorical=[]
print("Categorical Columns unique values")
for c in columns:
    if(df[c].dtype=='object'):
        categorical.append(c)
for i,c in enumerate(categorical):
    print(f"\n {i+1}. Column {c}\n    Unique Values : {df[c].unique()}\n")
    


# In[6]:


print("Split of data by each categorical values for each categorical features")
for i,c in enumerate(categorical):
    print(f"\n {i+1}. Column {c}\n    Unique Count : {df[c].value_counts(dropna=False)} ")


# ### Missing Value Analysis

# In[7]:


print("Null values present in each feature")
df.isna().sum()


# #### From above, its evident that there are missing values in the dataset for 3 features : workclass, occupation and native-country 

# ### Detecting Missing value category for Workclass

# In[8]:


df[df['workclass'].isna()].head(10)


# In[9]:


print("Split up of missing workclass data by native country")
df[df['workclass'].isna()]['native-country'].value_counts()


# #### We can't identify any solid pattern for missing data, here majority of data missing belongs to people whose native-country is USA but that could be since training dataset does have majority of cases from United-States <br>Hence the workclass data is MCAR(missing completely at random)

# In[10]:


print("Majority of data has native-country as USA")
df['native-country'].value_counts()


# ### Detecting Missing value category for Occupation

# In[11]:


df[df['occupation'].isna()]['workclass'].unique()


# In[12]:


df[~df['occupation'].isna()]['workclass'].unique()


# In[20]:


# Assuming df is your DataFrame
descriptive_stats = df.describe(include=object)

# Adding a label
descriptive_stats.style.set_caption("Descriptive Statistics for Categorical Variables")


# In[23]:


import missingno as msno


# In[24]:


msno.matrix(df)


# In[25]:


msno.bar(df) 


# In[26]:


msno.heatmap(df) 


# In[2]:


print("From the heatmap it is evident that Occupation with non missing values have workclass values defined\nOccupation seems to be missing whenever the Workclass field is absent and hence is NMAR (not missing at random)")


# In[27]:


total_rows_with_missing_values = df.isna().any(axis=1).sum()


# In[28]:


print("Total rows having missing value",total_rows_with_missing_values)


# In[30]:


print("Percentage of missing data")
print(total_rows_with_missing_values/len(df)*100)


# 

# ### Detecting Missing value category for Native Country

# In[13]:


print("Checking split by target variable")
print(df[df['native-country'].isna()]['income_>50K'].value_counts())
print("Although majority of the cases are related to class 0, but that also could be since the distribution itself is skewed towards class 0. No conclusive evidence can be inferred")


# In[14]:


print("Checking overview to see any potential pattern")
df[df['native-country'].isna()].head()


# In[15]:


print("Checking by the race column")
df_n = df.copy()

# Replace NaN with a label (e.g., 'Unknown') in the copied DataFrame
df_n['native-country'].fillna('Unknown', inplace=True)
race_split_by_country = df_n.groupby('native-country')['race'].value_counts(dropna=False)

print("Split of 'race' grouped by 'native-country':")
print(race_split_by_country)


# #### From analysis above, it seems there is no clear pattern for missing native country and hence its MCAR (missing completely at random)

# ### Checking the split of data by target variable (Income >50k)

# In[16]:


colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
plt.figure(figsize=(8, 8))
    
df['income_>50K'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90,labels=["< 50K",">50K"],radius=0.8,colors=colors)

plt.title('Percentage Split of Data by target variable (Income Class)')
plt.ylabel('')
plt.subplots_adjust(top=0.45) 
plt.show()


# #### Data is imbalanced with majority of the observed records  having income value less than 50K

# In[17]:


df.hist(figsize=(12, 8), bins=10, edgecolor='black')
plt.suptitle('Histograms', x=0.5, y=1.02, ha='center', fontsize='x-large')
plt.show()


# #### From the histogram it is evident that the data is not normally distributed and its skewed. We also observe that capital-gain and capital-loss have majority of their values close to zero but there are very few counts having huge variation which warrants further investigation.

# In[18]:


print("Boxplot showing educational num vs workclass")
sns.boxplot(x='workclass',y='educational-num',data=df)
plt.xticks(rotation=90)


# #### Data seems to have almost similar trend for almost all workclass. There seems to be quite few outliers in the private sector which requires further investigation as to how it might impact the model.

# ### Show three example items from the data set in detail.

# In[19]:


seed = 42

random_sample = df.sample(n=3, random_state=seed)

print("Randomly selected three items from the dataset:")
random_sample


# #### These details provide information about the demographic attributes, employment details, and income status for each individual. <br>The first individual is a 43-year-old married male working in the Local Government sector with a high capital loss and a workweek of 40 hours, earning less than or equal to 50,000.<br><br> The 2nd individual is a 59-year-old unmarried female working in the private sector. With a low educational level (9th grade) and an educational-num of 5, she is widowed and works in the service industry. Despite no capital gain or loss, she works 40 hours per week and earns less than or equal to 50,000. <br><br> The third individual is a 29-year-old married female with a Master's degree working in the local government sector. As a professional in a specialty occupation, she is the wife in a married-civilian-spouse relationship. With no capital gain or loss, she works 40 hours per week and earns more than $50,000 annually.

# In[ ]:




