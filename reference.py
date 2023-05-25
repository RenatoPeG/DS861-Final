#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS 861 - Prof Minh Pham - Spring 2023
Final Exam
Name: Saksham Motwani
SFSU ID: 922988440
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore') 

data = pd.read_csv('vaccine response.csv') # Dimensions - 415 rows and 40 columns

# %%

# 1 - DATA PREPROCESSING

# Looking at the datatypes of each column.
print(data.info())

# Columns 'EGF', 'GRO', 'IL.1RA', and 'TNF.a' have some values which are non-numeric (thats why these columns are object type)
# Column 'EGF' has 1 instance with value 'Sample unable to be located so no data available for this timepoint'
# Column 'GRO' has 2 instances with values '> 10000'
# Column 'IL.1RA' has 10 instances with values '> 10000'
# Column 'TNF.a' has 1 instances with value '< 3.2'
# First lets convert all these values into missing values. This is strictly as per instruction in the question document and as per my understanding of the clarification email I sent to the professor on the day of the exam.
data.replace('Sample unable to be located so no data available for this timepoint', np.nan, inplace = True)
data.replace('> 10000', np.nan, inplace = True)
data.replace('< 3.2', np.nan, inplace = True)

# Print the number of missing values in each column
print(data.isnull().sum())

# Converting 'EGF', 'GRO', 'IL.1RA', and 'TNF.a' columns to type 'float64'
data['EGF'] = data['EGF'].astype('float64')
data['GRO'] = data['GRO'].astype('float64')
data['IL.1RA'] = data['IL.1RA'].astype('float64')
data['TNF.a'] = data['TNF.a'].astype('float64')

# Imputing each missing value with the minimum value of the corresponding column
data.fillna(data.min(), inplace = True)

# Check for missing values again - Should be 0 in all columns
print(data.isnull().sum())

# Removing instances where Age is greater than 120 days
data.drop(data[data['Age'] > 120].index, inplace = True)
data.reset_index(drop = True, inplace = True)

# Dimensions of final dataset (209 rows and 40 columns)
print('Dimensions of processed dataset', data.shape)


# %%

# 2 - INFERENCES

# Since we need to only consider LVR and HVR in the response variable, I will create a new dataframe
# and drop values of NVR. Then build a logistic regression model on that dataframe and proceed toward inference.

# Creating a copy of the preprocessed dataset
dataset_LR = data.copy()

# Removing instances where the value of the VR column is NVR
dataset_LR.drop(dataset_LR[dataset_LR['VR'] == 'NVR'].index, inplace = True)
dataset_LR.reset_index(drop = True, inplace = True)

# Creating dummy for the response variable. The reference value I have chosen is LVR.
VR_dummy = pd.get_dummies(dataset_LR['VR'], prefix = 'VR')
del VR_dummy['VR_LVR']

dataset_LR = pd.concat([dataset_LR, VR_dummy], axis = 1)

# Deleting the unnecessary column
del dataset_LR['VR']

# Predictors
X_LR = dataset_LR.copy()
X_LR.drop('VR_HVR', axis = 1, inplace = True)

# Response
y_LR = dataset_LR['VR_HVR']

# Adding the intercept manually
X_LR_int = sm.add_constant(X_LR)

# Logistic Regression using statsmodes
logit = sm.Logit(y_LR, X_LR_int).fit()
logit.summary()

# Significant variables (those with p value <= 0.05) & their interpretations

# Variable    p-value      Coefficient    Interpretation
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# EGF          0.050         -0.0461      exp(-0.0461) = 0.9549            --> For every 1 unit increase in EGF, the odds of having HVR decrease by approximately 4.51% compared to the odds of having LVR, while holding all other predictors constant.
# Eotaxin      0.024         -4.6950      exp(-4.6950) = 0.0091            --> For every 1 unit increase in Eotaxin, the odds of having HVR decrease by approximately 99.09% compared to the odds of having LVR, while holding all other predictors constant.
# Flt.3L       0.039         -2.4541      exp(-2.4541) = 0.0859            --> For every 1 unit increase in Flt.3L, the odds of having HVR decrease by approximately 91.41% compared to the odds of having LVR, while holding all other predictors constant.
# IFN.g        0.024         10.3082      exp(10.3082) = 29977.4287        --> For every 1 unit increase in IFN.g, the odds of having HVR increase by approximately 2997643% compared to the odds of having LVR, while holding all other predictors constant.
# MCP.3        0.038          2.5857      exp(2.5857)  = 13.2726           --> For every 1 unit increase in MCP.3, the odds of having HVR increase by approximately 1227% compared to the odds of having LVR, while holding all other predictors constant.
# IL.2         0.005         37.4197      exp(37.4197) = 16133915435063100 --> For every 1 unit increase in IL.2, the odds of having HVR increase by approximately 1613391543506309900% compared to the odds of having LVR, while holding all other predictors constant.
# IL.4         0.028          0.5547      exp(0.5547)  = 1.7414            --> For every 1 unit increase in IL.4, the odds of having HVR increase by approximately 74% compared to the odds of having LVR, while holding all other predictors constant.

# %%

# 3 - BAGGING TREE, RANDOM FOREST, and BOOSTING TREE


# We need to encode the 3 classes of the response variable  so they can be understood by the machine
# Mapping NVR as 1, LVR as 2, and HVR as 3
encoding_dict = {'NVR': 1, 'LVR': 2, 'HVR': 3}
data['VR'].replace(encoding_dict, inplace = True)

X = data.copy()
X.drop('VR', axis = 1, inplace = True)
y = data['VR']

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.3, random_state = 1000)

# %%
# BAGGING with 5-Fold CV to TUNE THE 2 PARAMETERS: n_estimators and max_depth
bagging_grid = {'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [2, 4, 6, 8, 10]}

bg_clf = GridSearchCV(RandomForestClassifier(min_samples_leaf = 5, max_features = None, random_state = 1000), param_grid = bagging_grid, cv = 5, scoring = 'accuracy', n_jobs = -1)
bg_clf.fit(X_train, y_train)
y_pred = bg_clf.predict(X_test)

print('Bagged Tree - Best Hyperparameters:', bg_clf.best_params_)
#print('Bagged Tree - F1 Score on Testing Set:', f1_score(y_test, y_hat))
#print('Bagged Tree - Confusion Matrix for Testing Set:\n', confusion_matrix(y_test, y_hat, labels = [1,0]))
print('Bagged Tree - Accuracy of Testing Set:', accuracy_score(y_test, y_pred))

# Refit the model to get important features
refit_bg_clf = RandomForestClassifier(n_estimators = bg_clf.best_params_['n_estimators'], 
                                      max_depth = bg_clf.best_params_['max_depth'],
                                      max_features = None,
                                      min_samples_leaf = 5, random_state = 1000)
refit_bg_clf.fit(X_train, y_train)
importance = pd.DataFrame({'feature':X.columns.values, 'importance':refit_bg_clf.feature_importances_})
print('Bagged Tree - Features ranked according to their importance are as below: \n')
print(importance[importance['importance'] != 0].sort_values(by = ['importance'], ascending = False))

"""
Bagged Tree - Best Hyperparameters: {'max_depth': 4, 'n_estimators': 300}
Bagged Tree - Accuracy of Testing Set: 0.47619047619047616
Bagged Tree - Features ranked according to their importance are as below: 

        feature  importance
25         IL.2    0.182170
22        IL.1a    0.059303
21       IL.1RA    0.054225
12        IL.10    0.041593
37        TNF.b    0.039416
16  IL.12..p70.    0.034833
0           Age    0.033970
26         IL.3    0.033738
32        IP.10    0.033110
27         IL.4    0.031702
28         IL.5    0.031042
23         IL.9    0.028593
20       IL.17A    0.027375
5         G.CSF    0.025458
15          MDC    0.021934
19       sCD40L    0.020728
8   Fractalkine    0.020517
3       Eotaxin    0.018755
14  IL.12..p40.    0.017920
29         IL.6    0.017833
17        IL.13    0.017566
34       MIP.1a    0.017040
36        TNF.a    0.016806
18        IL.15    0.015181
35       MIP.1b    0.014843
10        IFN.g    0.013859
1           EGF    0.013269
4         TGF.a    0.012017
11          GRO    0.011871
31         IL.8    0.010945
6        Flt.3L    0.010888
38         VEGF    0.010849
7        GM.CSF    0.010774
30         IL.7    0.010199
2         FGF.2    0.010081
24        IL.1b    0.010036
13        MCP.3    0.008807
33        MCP.1    0.005403
9        IFN.a2    0.005352
"""

# %%

# BAGGING with 5-Fold CV to TUNE THE 2 PARAMETERS: n_estimators and max_depth FOR A BAGGING TREE
