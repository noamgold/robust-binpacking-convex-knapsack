import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import math
from sklearn.metrics import explained_variance_score # Import explained_variance_score
from sklearn.model_selection import train_test_split # Import train_test_split
from sklearn.metrics import mean_squared_error # Import mean_squared_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

df=pd.read_excel('Department 13300_English_Lean.xlsx')

# Separate features (X) and target variable (y)
y = df.iloc[:, 3]  # First column as the target variable
X = df.iloc[:, 4:] # Rest of the columns as features

#for col in X.columns:
#    X[col] = pd.to_numeric(X[col], errors='coerce')
#    X[col] = X[col].fillna(X[col].mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate the min_samples_split based on the training set size
min_samples_split = 10*math.ceil(math.sqrt(len(X_train)))

# Calculate the min_samples_leaf based on the training set size
min_samples_leaf = 3*math.ceil(math.sqrt(len(X_train)))

# Create the decision tree regressor with min_samples_leaf
dtree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, random_state=42)

# Create a KFold object with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize a list to store the R-squared values for each fold
r2_scores = []

# Loop through each fold
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Train the decision tree model on the training fold
    dtree.fit(X_train_fold, y_train_fold)

    # Evaluate the model on the validation fold and store the R-squared value
    r2 = dtree.score(X_val_fold, y_val_fold)
    r2_scores.append(r2)

# Calculate the average R-squared score across all folds
average_r2 = np.mean(r2_scores)
print("Average R-squared across 5-fold cross-validation:", average_r2)

# Fit the model to the entire training data
dtree.fit(X_train, y_train)

# Evaluate the model on the test data
test_r2 = dtree.score(X_test, y_test)
print("R-squared on the test data:", test_r2)


# Assuming dtree and X are already defined as in the previous code

plt.figure(figsize=(20,10))
plot_tree(dtree, filled=True, feature_names=X.columns, rounded=True)
plt.show()
