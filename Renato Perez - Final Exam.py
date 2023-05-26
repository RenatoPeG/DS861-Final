# %%

import warnings
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import statsmodels.api as sm
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# %%

print("""
# ================================================
# 1. PREPROCESSING
# ================================================
""")

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

print("""
# ================================================
# 2. INFERENCES
# ================================================ """)

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

print("""

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

print("""
# ================================================
# 3. TREES
# ================================================ """)

# Encoding response variable
encoding = {"LVR": 1, "NVR": 2, "HVR": 3}
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

# Defining Grid Search Hyperparams
bagging_gs = {"n_estimators": [100, 150, 200, 250, 300, 350, 400, 450, 500],
              "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10]}

# Grid Search
bg_clf_train = GridSearchCV(
    RandomForestClassifier(
        min_samples_leaf=10,
        max_features=None,
        random_state=1
    ),
    param_grid=bagging_gs,
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
Test Accuracy: {round(accuracy_score(y_test, y_pred), 4)}
"""
)

# Refit model with best hyperparams
bg_clf_best = RandomForestClassifier(
    **bg_clf_train.best_params_,
    max_features=None,
    min_samples_leaf=10,
    random_state=1,
)

# Fit the model
bg_clf_best.fit(X_train, y_train)

# Feature importances
def feature_importances(best_model):
    importance = pd.DataFrame(
        {"feature": X.columns.values, "importance": best_model.feature_importances_}
    )

    print(
        f"""Features according to Importances
    {importance[importance["importance"] != 0].sort_values(
        by="importance", ascending=False)}
    """
    )

feature_importances(bg_clf_best)

trees = [["Bagging", round(accuracy_score(y_test, y_pred), 4)]]


# %%%

# RANDOM FOREST
# -------------

# Defining Grid Search Hyperparams
random_gs = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500],
           'max_depth': [2, 4, 6, 8, 10, 12, 14, 16, 18]}

# Grid Search
rnd_clf_train = GridSearchCV(
    RandomForestClassifier(
        min_samples_leaf=10, max_features=None, random_state=1),
    param_grid=random_gs,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

# Fit the model
rnd_clf_train.fit(X_train, y_train)

# Predicting on test set
y_pred = rnd_clf_train.predict(X_test)

print(
    f"""
RANDOM FOREST
-------------

Best Hyperparamaters: {rnd_clf_train.best_params_}
Test Accuracy: {round(accuracy_score(y_test, y_pred), 4)}
"""
)

# Refit model with best hyperparams
rnd_clf_best = RandomForestClassifier(
    **rnd_clf_train.best_params_,
    max_features=None,
    min_samples_leaf=10,
    random_state=1,
)

# Fit the model
rnd_clf_best.fit(X_train, y_train)

# Feature importances
feature_importances(rnd_clf_best)

trees.append(["Random Forest", round(accuracy_score(y_test, y_pred), 4)])


# %%%

# BOOSTING
# --------

# Defining Grid Search Hyperparams
boosting_gs = {"n_estimators": [100, 200, 300, 400, 500],
               "learning_rate": [0.01, 0.1, 1.0]}

# Grid Search
bst_clf_train = GridSearchCV(
    GradientBoostingClassifier(
        min_samples_leaf=5, random_state=1),
    param_grid=boosting_gs,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

# Fit the model
bst_clf_train.fit(X_train, y_train)

# Predicting on test set
y_pred = bst_clf_train.predict(X_test)

print(
    f"""
BOOSTING
--------

Best Hyperparamaters: {bst_clf_train.best_params_}
Test Accuracy: {round(accuracy_score(y_test, y_pred), 4)}
"""
)

# Refit model with best hyperparams
bst_clf_best = GradientBoostingClassifier(
    **bst_clf_train.best_params_,
    min_samples_leaf=5,
    random_state=1,
)

# Fit the model
bst_clf_best.fit(X_train, y_train)

# Feature importances
feature_importances(bst_clf_best)

trees.append(["Boosting", round(accuracy_score(y_test, y_pred), 4)])


# %%%

summary = pd.DataFrame(trees, columns=["Tree", "Accuracy"])

print(f"""
SUMMARY
-------

{summary}

The best model to select based on accuracy is the Boosting Tree model.
""")


# %%

print("""
# ================================================
# 4. PCA
# ================================================
""")

# (a) Reload the original data set, impute missing values, and
# drop the Age and VR variables.
# ------------------------------------------------------------

# Loading data
df_pca = pd.read_csv("vaccine response.csv")

# Replace the string "> 10000" with NaN in the "GRO" and
# "IL.1RA" columns
df_pca["GRO"] = df_pca["GRO"].replace("> 10000", np.nan)
df_pca["IL.1RA"] = df_pca["IL.1RA"].replace("> 10000", np.nan)
df_pca["TNF.a"] = df_pca["TNF.a"].replace("< 3.2", np.nan)

# Convert the affected columns to numeric type (float)
df_pca["EGF"] = pd.to_numeric(df_pca["EGF"], errors="coerce")
df_pca["GRO"] = pd.to_numeric(df_pca["GRO"], errors="coerce")
df_pca["IL.1RA"] = pd.to_numeric(df_pca["IL.1RA"], errors="coerce")
df_pca["TNF.a"] = pd.to_numeric(df_pca["TNF.a"], errors="coerce")

# Impute missing values with the minimum value of each column
df_pca.fillna(df_pca.min(), inplace=True)

# Drop variables Age and VR
X = df_pca.copy()
X = X.drop(["Age", "VR"], axis=1)
y = df_pca["VR"]


# %%%

# (b) Perform PCA and plot the scree plot.
# ------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Instantiate and fit PCA
pca = PCA()
pca.fit(X_scaled)

# Explained Variance Ratio
evr = pca.explained_variance_ratio_

# EVR Scree plot
plt.plot(range(1, len(evr)+1), evr)
plt.xlabel("Number of components")
plt.ylabel("Proportion Variance Explained")
plt.title("EVR Scree Plot")
plt.show()

# Cumulative EVR Scree plot
plt.plot(range(1, len(evr)+1), np.cumsum(evr))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Proportion Variance Explained")
plt.title("Cumulative EVR Scree Plot")
plt.show()

print("""
Looks like there is not a clear elbow in the cumulative plot. The 
variability is spread out evenly across the predictor space.
""")

# %%%

# (c) How many principal components do you need to cover 90%
# variability?
# ------------------------------------------------------------

pca = PCA(0.9)
pca.fit(X_scaled)
X_trans = pca.transform(X_scaled)

print(f"""
Principal Components for 90% Variability:  {X_trans.shape[1]}

It went from {X.shape[1]} to {X_trans.shape[1]} components.""")


# %%%

# (d) Plot the data set using the first two principal component
# PC1 and PC2. Color the data point using a color code of your choice.
# ------------------------------------------------------------

# Colors LVR=red, NVR=green, HVR=blue
colors = ListedColormap(["#EA526F", "#51A3A3", "#235789"])

# Enconding the response variables to a numeric value
enc = {"LVR": 0, "NVR": 1, "HVR": 2}
y = y.replace(enc)

# Plot
scatter = plt.scatter(X_trans[:, 0], X_trans[:, 1], c=y, cmap=colors, alpha=0.65)
plt.legend(handles=scatter.legend_elements()[0], labels=list(enc.keys()))
plt.grid(alpha=0.1)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("First 2 PCs by VR")
plt.show()

print("It appears that there is not a clearly defined separation \nbetween the classes.")


# %%

print("""
# ====================================
# EXTRA CREDIT
# ====================================

# Build a lasso logistic regression model using all features
# of cytokines and all observations.

# This includes all steps of training-validating-testing split,
# tune parameter…Is there any variable
# (cytokine) that stands out?
# ------------------------------------------------------------ """)


def load_df_extra():
    df_extra = pd.read_csv("vaccine response.csv")

    # Convert specific columns to numeric, replacing non-numeric values with NaN
    df_extra['TNF.a'] = pd.to_numeric(df_extra['TNF.a'], errors='coerce')
    df_extra['EGF'] = pd.to_numeric(df_extra['EGF'], errors='coerce')
    df_extra['GRO'] = pd.to_numeric(df_extra['GRO'], errors='coerce')
    df_extra['IL.1RA'] = pd.to_numeric(df_extra['IL.1RA'], errors='coerce')

    # Drop rows with NaN values
    return df_extra.dropna()


def lr_l1(df_extra):
    # Assign X and y
    X = df_extra.drop(columns=['VR'])  # Features
    y = df_extra['VR']  # Target variable

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    # Scale the train data
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))

    # GridSearch 5-Fold CV
    param_grid = {'C': np.logspace(-10, 10, 21)}
    lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=1)
    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Best model Train set
    lr_lasso = LogisticRegression(
        penalty='l1', solver='liblinear', **grid_search.best_params_, random_state=1)
    lr_lasso.fit(X_train, y_train)

    # Train Accuracy
    y_train_pred = lr_lasso.predict(X_test)
    valid_accuracy = accuracy_score(y_test, y_train_pred)

    # Scale Test set
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test = pd.DataFrame(scaler.transform(X_test))

    # Best model Test set
    lr_lasso = LogisticRegression(
        penalty='l1', solver='liblinear', **grid_search.best_params_, random_state=1)
    lr_lasso.fit(X_train, y_train)

    # Test Accuracy
    y_test_pred = lr_lasso.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Significant features using Lasso regularization
    lasso_coefs = lr_lasso.coef_[0]
    idxs = np.argsort(np.abs(lasso_coefs))[::-1][:10]
    sig_features_t10 = X.columns[idxs]
    sig_features_t10 = pd.DataFrame(list(lasso_coefs[idxs]), index=list(
        X.columns[idxs]), columns=["Coefficient"])

    print(f"""
{'-'.join(df_extra['VR'].value_counts().index.to_list())}
-------

Train Set
    Best C: {grid_search.best_params_["C"]}
    Accuracy: {round(valid_accuracy, 4)}

Test Set
    Accuracy: {round(test_accuracy, 4)}

Top 10 Significant Features
{sig_features_t10}

""")

# (a) Consider a binary classification problem with response
# variable of values LVR and HVR.
# ------------------------------------------------------------


print("(a)")
# Drop 'NVR' from the 'VR' column
df_extra = load_df_extra()
df_extra = df_extra[df_extra['VR'] != 'NVR']
lr_l1(df_extra)

# (b) Repeat the previous part but with response variable of
# values HVR and NVR.
# ------------------------------------------------------------

print("(b)")
# Drop 'LVR' from the 'VR' column
df_extra = load_df_extra()
df_extra = df_extra[df_extra['VR'] != 'LVR']
lr_l1(df_extra)

print("""SUMMARY
-------

In scenario (a), where NVR is dropped and the classification task 
becomes binary between LVR and HVR, the logistic regression model 
achieves an accuracy of 0.5 on both the train and test sets. This 
accuracy indicates that the model is performing at chance level, 
meaning it's not able to effectively distinguish between the two 
classes. 

In scenario (b), where LVR is dropped and the classification task 
becomes binary between NVR and HVR, the logistic regression model 
achieves an accuracy of 0.1538 on both the train and test sets. 
Similarly to scenario (a), this accuracy is also indicative of poor 
performance, as the model is not able to effectively distinguish 
between the two classes. 
""")

# %%
