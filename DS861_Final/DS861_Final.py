#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 08:32:16 2023

@author: stevetruong
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

#%%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 10)
df = pd.read_csv("/Users/stevetruong/Desktop/DS861 Final/vaccine response.csv")
df
#%%
#1. Preprocess data: There are some missing values in the data set. For each missing value, you can impute
# that value with the MINIMUM value of the corresponding column. Remove all children that are older than
# 120 days. Report the dimension of the final data set.(10 pts)

# *** Convert columns to numeric *** because these columns contain strings likely due to input error,
# the decision was made to indiscriminitely consider them NaN's rather than try to interpret***
df['TNF.a'] = pd.to_numeric(df['TNF.a'], errors='coerce')
df['EGF'] = pd.to_numeric(df['EGF'], errors='coerce')
df['GRO'] = pd.to_numeric(df['GRO'], errors='coerce')
df['IL.1RA'] = pd.to_numeric(df['IL.1RA'], errors='coerce')

# Impute missing values with the minimum value of each column
df = df.fillna(df.min())

# Remove children older than 120 days
df = df[df['Age'] <= 120]

# Report the dimensions of the final dataset
print(f'Final dataset dimensions:{df.shape}')

#%%
# 2. Inferences(10 pts )
# Consider a binary classification problem with response variable of values LVR and HVR.  Build a logistic 
# regression model using all features of cytokines and all observations. Provide a list of significant
# variables and interpret them. 
# Create dummy variables for the response variable

df['Response'] = pd.get_dummies(df['VR'])['HVR']

# Define the features and the target variable
X = df.drop(['VR', 'Response'], axis=1)
y = df['Response']

# Add constant term
features = sm.add_constant(X)

# Fit the logistic regression model
logit_model = sm.Logit(y, features)
result = logit_model.fit()

# Get the summary of the model
print(result.summary())

# Get the significant variables with p-value < 0.05
significant_variables = result.pvalues[result.pvalues < 0.05].index

# Sort the coefficients by magnitude
sorted_coefficients = result.params.reindex(significant_variables).abs().nlargest(6)

print("Top 5 Significant Coefficients:")
for variable, coefficient in sorted_coefficients.iteritems():
    sign = "+" if result.params[variable] >= 0 else "-"
    print(f"{variable}\tCoefficient: {sign}{coefficient:.4f}")


#############################Interpretations for the five largest coefficients with p-values of < 0.05:#############################
# ***Intepretation assumes all other variables are held constant.

# IL.2: The coefficient of IL.2 is +7.4628, indicating that a one-unit increase in IL.2 is associated with an 
# increase of approximately 7.4628 in the predicted probability of having a high VR (HVR)
# IL.15: The coefficient of IL.15 is +2.7220, suggesting that a one-unit increase in IL.15 is associated with an 
# increase of approximately 2.7220 in the predicted probability of having a high VR (HVR)
# IL.17A: The coefficient of IL.17A is -0.9419, indicating that a one-unit increase in IL.17A is associated 
# with a decrease of approximately 0.9419 in the predicted probability of having a high VR (HVR)
# M.CSF: The coefficient of GM.CSF is -0.9209, suggesting that a one-unit increase in GM.CSF is associated with a 
# decrease of approximately 0.9209 in the predicted probability of having a high VR (HVR)
# IL.4: The coefficient of IL.4 is +0.2195, indicating that a one-unit increase in IL.4 is associated with an
# increase of approximately 0.2195 in the predicted probability of having a high VR (HVR)
#%%
# 3.(30pts) Build a bagging tree, a random forest, and a boosting tree model to make predictions of response
# variables of all 3 values: LVR, NVR, and HVR. Perform prediction on the testing set. Report the accuracy 
# score. Report the important variables for each model. 
     #You are required to tune at least 2 hyperparameters for each model.
     #You are required to perform hyperparameters tuning using 5-fold cross validation.
     #select your best model using the accuracy score.

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using 5-fold cross-validation

# Bagging Tree
bagging_params = {'n_estimators': [50, 100, 150], 'max_samples': [0.5, 0.7, 0.9]}
bagging_grid = GridSearchCV(BaggingClassifier(), bagging_params, cv=5)
bagging_grid.fit(X_train, y_train)
best_bagging = bagging_grid.best_estimator_

# Random Forest
rf_params = {'n_estimators': [50, 100, 150], 'max_depth': [None, 5, 10]}
rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_


# Boosting Tree
# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

boosting_params = {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 1.0]}
boosting_grid = GridSearchCV(GradientBoostingClassifier(), boosting_params, cv=5)
boosting_grid.fit(X_train_scaled, y_train)
best_boosting = boosting_grid.best_estimator_

# Train the models using the best hyperparameters
best_bagging.fit(X_train, y_train)
best_rf.fit(X_train, y_train)
best_boosting.fit(X_train, y_train)

# Make predictions on the testing set and calculate accuracy score
bagging_pred = best_bagging.predict(X_test)
rf_pred = best_rf.predict(X_test)
boosting_pred = best_boosting.predict(X_test)

bagging_accuracy = accuracy_score(y_test, bagging_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
boosting_accuracy = accuracy_score(y_test, boosting_pred)

accuracies = {
    "Bagging Tree Accuracy": bagging_accuracy,
    "Random Forest Accuracy": rf_accuracy,
    "Boosting Tree Accuracy": boosting_accuracy
}

highest_accuracy = max(accuracies.values())
highest_accuracy_label = [label for label, accuracy in accuracies.items() if accuracy == highest_accuracy][0]

print("Bagging Tree Accuracy:", round(bagging_accuracy, 4))
print("Random Forest Accuracy:", round(rf_accuracy, 4))
print("Boosting Tree Accuracy:", round(boosting_accuracy, 4))
print("Best Model with Highest Accuracy:", highest_accuracy_label, round(highest_accuracy, 4))



#############################Chart to compare accuracy#############################

# Accuracy scores
models = ['Bagging Tree', 'Random Forest', 'Boosting Tree']
accuracies = [bagging_accuracy, rf_accuracy, boosting_accuracy]

# Plotting the bar chart
plt.bar(models, accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Models')
plt.ylim([0.0, 1.0])

# Displaying the values on top of each bar
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, str(round(v, 2)), ha='center')

# Display the graph
plt.show()

#############################Determine the important variables for each model#############################

# Bagging Tree Important Variables
bagging_estimators = best_bagging.estimators_
bagging_importance = np.mean([tree.feature_importances_ for tree in bagging_estimators], axis=0)
bagging_feature_names = X.columns

rf_importance = best_rf.feature_importances_
rf_feature_names = X.columns

# Boosting Tree Important Variables
boosting_importance = best_boosting.feature_importances_
boosting_feature_names = X.columns


# Sort the feature importance values and corresponding names in descending order
bagging_top_indices = np.argsort(bagging_importance)[::-1][:5]
bagging_top_variables = bagging_feature_names[bagging_top_indices]
bagging_top_importance = bagging_importance[bagging_top_indices]

# Sort the feature importance values and corresponding names in descending order
rf_top_indices = np.argsort(rf_importance)[::-1][:5]
rf_top_variables = rf_feature_names[rf_top_indices]
rf_top_importance = rf_importance[rf_top_indices]

boosting_top_indices = np.argsort(boosting_importance)[::-1][:5]
boosting_top_variables = boosting_feature_names[boosting_top_indices]
boosting_top_importance = boosting_importance[boosting_top_indices]


# Print the top five important variables for Bagging Tree
print("\nBagging Tree Important Variables:")
for variable, importance in zip(bagging_top_variables, bagging_top_importance):
    print(f"{variable}: {importance}")

# Print the top five important variables for Random Forest
print("\nRandom Forest Important Variables:")
for variable, importance in zip(rf_top_variables, rf_top_importance):
    print(f"{variable}: {importance}")

# Print the top five important variables for Boosting Tree
print("\nBoosting Tree Important Variables:")
for variable, importance in zip(boosting_top_variables, boosting_top_importance):
    print(f"{variable}: {importance}")


#############################CHart to compare the important variables#############################

# Random Forest
rf_top_variables = rf_feature_names[np.argsort(rf_importance)][::-1][:5]
rf_top_importance = rf_importance[np.argsort(rf_importance)][::-1][:5]

# Boosting Tree
boosting_top_variables = boosting_feature_names[np.argsort(boosting_importance)][::-1][:5]
boosting_top_importance = boosting_importance[np.argsort(boosting_importance)][::-1][:5]

# Bagging Tree
bagging_top_variables = bagging_feature_names[np.argsort(bagging_importance)][::-1][:5]
bagging_top_importance = bagging_importance[np.argsort(bagging_importance)][::-1][:5]

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.barh(rf_top_variables, rf_top_importance, align='center', color='blue', label='Random Forest')
plt.barh(boosting_top_variables, boosting_top_importance, align='center', color='green', label='Boosting Tree')
plt.barh(bagging_top_variables, bagging_top_importance, align='center', color='orange', label='Bagging Tree')

plt.xlabel('Importance')
plt.ylabel('Variables')
plt.title('Top Five Important Variables Comparison')
plt.legend()

plt.show()

#%%
# 4. PCA (20 pts)

# (a) Reload the original data set, impute missing values, and drop the Age and VR variables.
df2 = pd.read_csv("/Users/stevetruong/Desktop/DS861 Final/vaccine response.csv")

df2['TNF.a'] = pd.to_numeric(df2['TNF.a'], errors='coerce')
df2['EGF'] = pd.to_numeric(df2['EGF'], errors='coerce')
df2['GRO'] = pd.to_numeric(df2['GRO'], errors='coerce')
df2['IL.1RA'] = pd.to_numeric(df2['IL.1RA'], errors='coerce')

df2 = df2.fillna(df2.min())

df2 = df2.drop(['Age', 'VR'], axis=1)


# (b) Perform PCA and plot the scree plot.
# Scale the data and fit
scaler = StandardScaler()
df2_scaled = scaler.fit_transform(df2)

pca = PCA()
pca.fit(df2_scaled)

# Scree plot
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()# Perform PCA
pca = PCA(n_components=2)  # Set the number of components to 2
principal_components = pca.fit_transform(df2_scaled)


# (c) How many principal components do you need to cover 90% variability?
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(cumulative_variance_ratio >= 0.9) + 1

print("Number of Principal Components for 90% Variability:", num_components)


# (d) Plot the data set using the first two principal components PC1 and PC2. Color the data point using a 
# color code of your choice.

# Perform PCA
pca = PCA(n_components=2)  # Set the number of components to 2
principal_components = pca.fit_transform(df2_scaled)

# Create a DataFrame with the principal components
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Define color codes for PC1 and PC2
color_pc1 = 'blue'
color_pc2 = 'red'

# Plot the data points
plt.scatter(principal_df['PC1'], principal_df['PC2'], c=[color_pc1 if x else color_pc2 for x in principal_df['PC1'] > 0])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Data Set - PC1 vs. PC2 (Color-coded)')
plt.show()




#%%
# Extra credit problem (optional)
# 5. (20pts) (a) Consider a binary classification problem with response variable of values LVR and HVR.  
# Build a lasso logistic regression model using all features of cytokines and all observations. 
# This includes all steps of training-validating-testing split, tune parameter…Is there any variable 
# (cytokine) that stands out?
# (b) Repeat the previous part but with response variable of values HVR and NVR.

#############################Part A: classification problem with response variable of values LVR and HVR################################
df3 = pd.read_csv("/Users/stevetruong/Desktop/DS861 Final/vaccine response.csv")

# Convert specific columns to numeric, replacing non-numeric values with NaN
df3['TNF.a'] = pd.to_numeric(df3['TNF.a'], errors='coerce')
df3['EGF'] = pd.to_numeric(df3['EGF'], errors='coerce')
df3['GRO'] = pd.to_numeric(df3['GRO'], errors='coerce')
df3['IL.1RA'] = pd.to_numeric(df3['IL.1RA'], errors='coerce')

# Drop rows with NaN values
df3 = df3.dropna()

# Drop 'NVR' from the 'VR' column
df3 = df3[df3['VR'] != 'NVR']

# Assign X and y
X = df3.drop(columns=['VR'])  # Features
y = df3['VR']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=92239484)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=922239484)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameter grid and perform hyperparameter tuning
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}

model = LogisticRegression(penalty='l1', solver='liblinear')
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
print("LVR and HVR Performance:")
print("Best hyperparameters:", grid_search.best_params_)

# Train the model
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', **grid_search.best_params_)
lasso_model.fit(X_train_scaled, y_train)

# Validate the model
y_val_pred = lasso_model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

# Test the model
y_test_pred = lasso_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)
print('\n')
# Identify significant features (cytokines) using Lasso regularization
lasso_coefs = lasso_model.coef_[0]
significant_cytokines = X.columns[lasso_coefs != 0]


# Get the five variables with the largest coefficients and their values
coef_indices = np.argsort(np.abs(lasso_coefs))[::-1][:5]
significant_variables = X.columns[coef_indices]
variable_coefficients = lasso_coefs[coef_indices]

print("LVR and HVR - Top 5 Significant Variables:")
for variable, coefficient in zip(significant_variables, variable_coefficients):
    print(f"{variable}\tCoefficient: {round(coefficient,4)}")
print('\n')


#############################Part B: classification problem with response variable of values HVR and NVR################################

df4 = pd.read_csv("/Users/stevetruong/Desktop/DS861 Final/vaccine response.csv")

# Convert specific columns to numeric, replacing non-numeric values with NaN
df4['TNF.a'] = pd.to_numeric(df4['TNF.a'], errors='coerce')
df4['EGF'] = pd.to_numeric(df4['EGF'], errors='coerce')
df4['GRO'] = pd.to_numeric(df4['GRO'], errors='coerce')
df4['IL.1RA'] = pd.to_numeric(df4['IL.1RA'], errors='coerce')

# Drop rows with NaN values
df4 = df4.dropna()

# Drop 'LVR' from the 'VR' column
df4 = df4[df4['VR'] != 'LVR']

# Assign X and y
X = df4.drop(columns=['VR'])  # Features
y = df4['VR']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=92239484)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=922239484)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameter grid and perform hyperparameter tuning
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}

model = LogisticRegression(penalty='l1', solver='liblinear')
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
print("________________________________________________________________________")
print("HVR and NVR Performance:")
print("Best hyperparameters:", grid_search.best_params_)

# Train the model
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', **grid_search.best_params_)
lasso_model.fit(X_train_scaled, y_train)

# Validate the model
y_val_pred = lasso_model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

# Test the model
y_test_pred = lasso_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)
print('\n')
# Identify significant features (cytokines) using Lasso regularization
lasso_coefs = lasso_model.coef_[0]
significant_cytokines = X.columns[lasso_coefs != 0]

# Get the five variables with the largest coefficients and their values
coef_indices = np.argsort(np.abs(lasso_coefs))[::-1][:5]
significant_variables = X.columns[coef_indices]
variable_coefficients = lasso_coefs[coef_indices]

print("HVR and NVR - Top 5 Significant Variables:")
for variable, coefficient in zip(significant_variables, variable_coefficients):
    print(f"{variable}\tCoefficient: {round(coefficient, 4)}")


################ Is there any variable (cytokine) that stands out? ################ 
# In the Lasso logistic regression models, IL.7 and TNF.b are specific to the HVR and NVR classification,
# while IFN.a2 and Age are specific to the LVR and HVR classification. This brings to attention the 
# variability of the model selection and feature importance techniques. The specific variables or 
# coefficients that stand out can vary depending on the modeling approach and the underlying data. 
# It's important to note that the results of the two Lasso should be used with caution. The decision to drop 
# data deemed as input error led to a large data loss. This led to problems such as high variability in results
# and potential bias in the model, calling to attention reliability issues.





