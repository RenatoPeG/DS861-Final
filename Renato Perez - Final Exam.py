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
df = pd.read_csv("vaccine response.csv")

# Removing missing values
# -----------------------

# Replace the string "> 10000" with NaN in the "GRO" and "IL.1RA" columns
df['GRO'] = df['GRO'].replace('> 10000', np.nan)
df['IL.1RA'] = df['IL.1RA'].replace('> 10000', np.nan)
df['TNF.a'] = df['TNF.a'].replace('< 3.2', np.nan)

# Convert the affected columns to numeric type (float)
df['EGF'] = pd.to_numeric(df['EGF'], errors='coerce')
df['GRO'] = pd.to_numeric(df['GRO'], errors='coerce')
df['IL.1RA'] = pd.to_numeric(df['IL.1RA'], errors='coerce')
df['TNF.a'] = pd.to_numeric(df['TNF.a'], errors='coerce')

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
logit.summary()


# %%

# ====================================
# 3. TREES
# ====================================


# %%

# ====================================
# 4. PCA
# ====================================


# %%
