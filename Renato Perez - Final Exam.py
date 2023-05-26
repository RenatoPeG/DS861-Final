# %%

import warnings
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
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

# Defining Grid Search Hyperparams
bagging_gs = {"n_estimators": [100, 200, 300, 400, 500],
              "max_depth": [2, 4, 6, 8, 10]}

# Grid Search
bg_clf_train = GridSearchCV(
    RandomForestClassifier(
        min_samples_leaf=5,
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
    min_samples_leaf=5,
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
random_gs = {"n_estimators": [100, 200, 300, 400, 500],
             "max_depth": [2, 4, 6, 8, 10]}

# Grid Search
rnd_clf_train = GridSearchCV(
    RandomForestClassifier(
        min_samples_leaf=5, max_features=None, random_state=1),
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
    min_samples_leaf=5,
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

# ====================================
# 4. PCA
# ====================================

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
df_pca = df_pca.drop(['Age', 'VR'], axis=1)


# %%%

# (b) Perform PCA and plot the scree plot.
# ------------------------------------------------------------

scaler = StandardScaler()
df_pca = scaler.fit_transform(df_pca)

# Instantiate and fit PCA
pca = PCA()
pca.fit(df_pca)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_pca)

# Scree plot
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()


# %%%

# (c) How many principal components do you need to cover 90%
# variability?
# ------------------------------------------------------------

cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(cumulative_variance_ratio >= 0.9) + 1

print("Principal Components for 90% Variability:", num_components)


# %%%

# (d) Plot the data set using the first two principal component
# PC1 and PC2. Color the data point using a color code of your
# choice (i.e LVR=red, NVR=green, HVR=blue).
# ------------------------------------------------------------

# Perform PCA
pca = PCA(n_components=2)  # Set the number of components to 2
principal_components = pca.fit_transform(df_pca)

# Create a DataFrame with the principal components
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Define color codes for PC1 and PC2
color_pc1 = 'blue'
color_pc2 = 'red'

# Plot the data points
plt.scatter(principal_df['PC1'], principal_df['PC2'], c=[
            color_pc1 if x else color_pc2 for x in principal_df['PC1'] > 0])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Data Set - PC1 vs. PC2')
plt.show()


# %%

# ====================================
# EXTRA CREDIT
# ====================================

# Build a lasso logistic regression model using all features
# of cytokines and all observations.

# This includes all steps of training-validating-testing split,
# tune parameter…Is there any variable
# (cytokine) that stands out?
# ------------------------------------------------------------


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

Performance:

The accuracy results indicate that scenario (b) (NVR vs. HVR) 
performed better on the test set compared to scenario (a) 
(LVR vs. HVR). This suggests that dropping the LVR class and 
performing binary classification between NVR and HVR led to a 
more effective model in this case.

It's important to note that the train set accuracy for scenario 
(b) was relatively low, indicating that the model might have 
struggled to capture the patterns in the training data. However, 
the model's generalization performance on the test set improved 
significantly, suggesting that it was able to identify relevant 
patterns and perform well on unseen data.

Significant Features:

Overlapping Features-. IL.8 and IL.17A appear as significant 
features in both scenarios. However, their coefficients and 
their associations with the classes are different. In scenario 
(a), IL.8 has a positive coefficient, indicating a positive 
association with the HVR class, while in scenario (b), IL.8 
also has a positive coefficient, suggesting its positive 
association with the NVR class.

Unique Features-. Some features are unique to each scenario. 
In scenario (a), features such as TNF.a, IL.12..p40., IFN.a2, 
MDC, sCD40L, MCP.3, and Eotaxin are significant, while in 
scenario (b), features like IFN.g, MIP.1b, MCP.1, IL.15, IL.7, 
TGF.a, and GM.CSF are significant.
""")
