# %%

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
import pandas as pd
import numpy as np


# %%

# ====================================
# 1. PREPROCESSING
# ====================================

# Loading data
df = pd.read_csv("vaccine response.csv")

# Removing missing values
# -----------------------

# Replace the string "> 10000" with NaN in the "GRO" and
# "IL.1RA" columns
df["GRO"] = df["GRO"].replace("> 10000", np.nan)
df["IL.1RA"] = df["IL.1RA"].replace("> 10000", np.nan)
df["TNF.a"] = df["TNF.a"].replace("< 3.2", np.nan)

# Convert the affected columns to numeric type (float)
df["EGF"] = pd.to_numeric(df["EGF"], errors="coerce")
df["GRO"] = pd.to_numeric(df["GRO"], errors="coerce")
df["IL.1RA"] = pd.to_numeric(df["IL.1RA"], errors="coerce")
df["TNF.a"] = pd.to_numeric(df["TNF.a"], errors="coerce")

# Impute missing values with the minimum value of each column
df.fillna(df.min(), inplace=True)


# The decision to replace the string values with NaN (Not a
# Number) is based on the assumption that these string values
# represent missing or unknown data. By replacing the string
# values with NaN, we can treat them as  missing values and
# handle them appropriately during df analysis or imputation.


# Filter out children older than 120 days
df = df[df["Age"] <= 120]

# Report the dimension of the final dataset
print("Dimension of the final data set:", df.shape)


# %%

# ====================================
# 2. INFERENCES
# ====================================

data = df.copy()

# Response variable of values LVR and HVR.
data = data[data["VR"] != "NVR"]
data["VR"] = data["VR"].apply(lambda x: 0 if x == "LVR" else 1)

# Predictors
X = data.copy()
X = X.drop(["VR"], axis=1)

# Response
y = data["VR"]

# Fit a logistic regression model using statsmodels.api
X_int = sm.add_constant(X)
logit = sm.Logit(y, X_int).fit()
# logit.summary()

# Results
# -------

# Significant variables (p <= 0.05) and interpretation

print(
    """
Interpretation of Results
-------------------------

Eotaxin: 
    *   The odds ratio for Eotaxin is exp(-4.6950) = 0.0091. 
    *   A one-unit increase in Eotaxin corresponds to a 99.09% 
        decrease n the odds of having HVR compared to LVR, 
        holding other variables constant.

Flt.3L: 
    *   The odds ratio for Flt.3L is exp(-2.4541) = 0.0859. 
    *   A one-unit increase in Flt.3L corresponds to an 91.41% 
        decrease in the odds of having HVR compared to LVR, 
        holding other variables constant.

IFN.g: 
    *   The odds ratio for IFN.g is exp(10.3082) = 29977.4287. 
    *   An increase in IFN.g by one unit leads to a 2,997,643.87% 
        increase in the odds of having HVR compared to LVR, 
        holding other variables constant.

MCP.3: 
    *   The odds ratio for MCP.3 is exp(2.5857) = 13.2726. 
    *   A one-unit increase in MCP.3 corresponds to a 1,227.26% 
        increase in the odds of having HVR compared to LVR, 
        holding other variables constant.

IL.2: 
    *   The odds ratio for IL.2 is exp(37.4197) = 1.7831 x 10^16. 
    *   An increase in IL.2 by one unit leads to a 1.7831 x 10^18% 
        increase in the odds of having HVR compared to LVR, 
        holding other variables constant.

IL.4: 
    *   The odds ratio for IL.4 is exp(0.5547) = 1.7414. 
    *   A one-unit increase in IL.4 corresponds to a 74.14% 
        increase in the odds of having HVR compared to LVR, 
        holding other variables constant.

IL.5: 
    *   The odds ratio for IL.5 is exp(3.1529) = 23.4038. 
    *   An increase in IL.5 by one unit leads to a 2,240.38% 
        increase in the odds of having HVR compared to LVR, 
        holding other variables constant.

IL.9: 
    *   The odds ratio for IL.9 is exp(-3.5983) = 0.0274. 
    *   A one-unit increase in IL.9 corresponds to a 97.26% 
        decrease in the odds of having HVR compared to LVR, 
        holding other variables constant.

IL.1a: 
    *   The odds ratio for IL.1a is exp(0.0738) = 1.0767. 
    *   An increase in IL.1a by one unit leads to a 7.67% 
        increase in the odds of having HVR compared to LVR, 
        holding other variables constant.

TNF.b: 
    *   The odds ratio for TNF.b is exp(6.1907) = 488.1877. 
    *   A one-unit increase in TNF.b corresponds to a 48,718.77% 
        increase in the odds of having HVR compared to LVR, 
        holding other variables constant.

VEGF: 
    *   The odds ratio for VEGF is exp(0.0736) = 1.0764. 
    *   An increase in VEGF by one unit leads to a 7.64% 
        increase in the odds of having HVR compared to LVR, 
        holding other variables constant.
"""
)


# %%

# ====================================
# 3. TREES
# ====================================

# Encoding response variable
encoding = {"NVR": 1, "LVR": 2, "HVR": 3}
df_trees = df.copy()
df_trees["VR"] = df_trees["VR"].replace(encoding)

# Predictors
X = df_trees.copy()
X = X.drop("VR", axis=1)

# Response
y = df_trees["VR"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%%

# BAGGING
# -------

"""
5-Fold CV
Hyperparameters:
    n_estimators
    max_depth
"""

# Defining Grid Search Hyperparams
bagging_sg = {"n_estimators": [100, 200, 300, 400, 500], "max_depth": [2, 4, 6, 8, 10]}

# Grid Search
bg_clf_train = GridSearchCV(
    RandomForestClassifier(min_samples_leaf=5, max_features=None, random_state=1),
    param_grid=bagging_sg,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

# Fit the model
bg_clf_train.fit(X_train, y_train)

# Predicting on test set
y_pred = bg_clf_train.predict(X_test)

print(
    f"""
BAGGING
-------

Best Hyperparamaters: {bg_clf_train.best_params_}
Accuracy: {round(accuracy_score(y_test, y_pred), 4)}
"""
)

# Refit model with best hyperparams
bg_clf_best = RandomForestClassifier(
    n_estimators=bg_clf_train.best_params_["n_estimators"],
    max_depth=bg_clf_train.best_params_["max_depth"],
    max_features=None,
    min_samples_leaf=5,
    random_state=1,
)

# Fit the model
bg_clf_best.fit(X_train, y_train)

# Feature importances
importance = pd.DataFrame(
    {"feature": X.columns.values, "importance": bg_clf_best.feature_importances_}
)

print(
    f"""Rank of Features according to Importances
{importance[importance["importance"] != 0].sort_values(by="importance", ascending=False)}
"""
)

# %%%

# RANDOM FOREST
# -------------


# %%%

# BOOSTING
# --------

# %%

# ====================================
# 4. PCA
# ====================================


# %%
