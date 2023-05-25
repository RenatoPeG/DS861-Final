# %%

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import pandas as pd
import numpy as np


# %%

# ====================================
# 1. PREPROCESSING
# ====================================

# Loading data
data = pd.read_csv("vaccine response.csv")

# Removing missing values
# -----------------------

# Replace the string "> 10000" with NaN in the "GRO" and "IL.1RA" columns
data['GRO'] = data['GRO'].replace('> 10000', np.nan)
data['IL.1RA'] = data['IL.1RA'].replace('> 10000', np.nan)
data['TNF.a'] = data['TNF.a'].replace('< 3.2', np.nan)

# Convert the affected columns to numeric type (float)
data['EGF'] = pd.to_numeric(data['EGF'], errors='coerce')
data['GRO'] = pd.to_numeric(data['GRO'], errors='coerce')
data['IL.1RA'] = pd.to_numeric(data['IL.1RA'], errors='coerce')
data['TNF.a'] = pd.to_numeric(data['TNF.a'], errors='coerce')

# Impute missing values with the minimum value of each column
data.fillna(data.min(), inplace=True)


# The decision to replace the string values with NaN (Not a
# Number) is based on the assumption that these string values
# represent missing or unknown data. By replacing the string
# values with NaN, we can treat them as  missing values and
# handle them appropriately during data analysis or imputation.


# Filter out children older than 120 days
data = data[data["Age"] <= 120]

# Report the dimension of the final dataset
print("Dimension of the final data set:", data.shape)


# %%

# ====================================
# 2. INFERENCES
# ====================================

# Response variable of values LVR and HVR.
data = data[data["VR"] != "NVR"]
data["VR"] = data["VR"].apply(lambda x: 0 if x == "LVR" else 1)

# Predictors
X = data.copy()
X = X.drop(["VR"], axis=1)

# Response
y = data["VR"]

# # Split holdout set
# X_train_valid, X_test, y_train_valid, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1)

# # Fit a logistic regression model using statsmodels.api. Interpret
# # coefficients that are significant.
# X_int = sm.add_constant(X_train_valid)

# import numpy as np
# from sklearn.model_selection import GridSearchCV

# # Set up the grid search parameters
# param_grid = {'alpha': np.logspace(-10, 10, 21)}

# # Define the logistic regression model
# logit = sm.Logit(y_train_valid, X_int)

# # Create the grid search object
# grid_search = GridSearchCV(logit, param_grid, cv=5)

# # Fit the grid search to find the best alpha value
# grid_search.fit(X_int, y_train_valid)

# # Get the best alpha value
# best_alpha = grid_search.best_params_['alpha']

# # Fit the logistic regression model with the best alpha value
# logit_reg = logit.fit_regularized(method='l1', alpha=best_alpha, maxiter=100)


# logit.summary()

"""
Perform a 5-fold cross-validation on the training test to choose the
best threshold for prediction by maximizing accuracy.
"""

# Split holdout set
X_train_valid, X_test, y_train_valid, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Split data into 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Candidate threshold values
thresholds = np.linspace(0, 1, 100)

# Define variables to keep track of best parameter and score
best_accuracy = {"Accuracy": float('-inf'), "Threshold": 0}

for train_index, valid_index in kf.split(X_train_valid):
    # Split data into train and validation sets for this fold
    X_train, X_valid = X_train_valid.iloc[train_index], X_train_valid.iloc[valid_index]
    y_train, y_valid = y_train_valid.iloc[train_index], y_train_valid.iloc[valid_index]

    # Define the logistic regression model with cross-validation
    clf = LogisticRegression(solver="newton-cg",
                             penalty="none", max_iter=10000, random_state=1)

    # Fit the logistic regression model on the training data
    clf.fit(X_train, y_train)

    # Make prediction on valid set
    y_pred = clf.predict_proba(X_valid)[:, 1]

    for a in thresholds:
        # Assigning the correct class based on the probability
        y_hat = np.where(y_pred > a, "LVR", "HVR")

        # Calculate accuracy
        accuracy = accuracy_score(y_valid, y_hat)

        if accuracy > best_accuracy["Accuracy"]:
            best_accuracy = {"Accuracy": accuracy, "Threshold": a}

print(f"""
TRAIN-VALID SET
---------------
    Best Accuracy: {round(best_accuracy["Accuracy"] * 100, 2)}
    Best Threshold: {round(best_accuracy["Threshold"], 4)}""")

"""
Fit Test set model with the best accuracy.
"""

# Create a Logistic Regression classifier with the optimal C value found using criteria
lr = LogisticRegression(solver="newton-cg", penalty="none",
                        max_iter=10000, random_state=1)

# Fit the classifier using the training + validation set
lr.fit(X_train_valid, y_train_valid)

# Get the predicted probabilities for the holdout set
y_pred = lr.predict_proba(X_test)[:, 1]

# Use the best threshold found using criteria to classify the holdout set
y_hat = np.where(y_pred > best_accuracy["Threshold"], "LVR", "HVR")

# Compute the confusion matrix for the holdout set using the FPR-based threshold
conf_mat = confusion_matrix(y_test, y_hat)

TP = conf_mat[1, 1]
FN = conf_mat[1, 0]
FP = conf_mat[0, 1]
TN = conf_mat[0, 0]

accuracy = accuracy_score(y_test, y_hat)
precision = TP / (TP + FP)

print(f"""

TEST SET
--------
    Accuracy: {round(accuracy * 100, 2)}%
    Precision: {round(precision * 100, 2)}%""")





# %%

# ====================================
# 3. TREES
# ====================================


# %%

# ====================================
# 4. PCA
# ====================================


# %%
