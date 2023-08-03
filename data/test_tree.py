#import time

import numpy as np
from numba.typed import List
from numba import jit, config
#config.DISABLE_JIT = True
import numpy as np
import pandas as pd


from rulefit import RuleFit
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import cross_val_score

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

AHAT_PTILE = 98

#rambam_data = pd.read_csv("OperatingRoom13300Merged.csv")
#rambam_data = pd.read_csv("OperatingRoom13200Ready.csv")
rambam_data = pd.read_excel("Department 13300_English.xlsx")

rambam_data = rambam_data[rambam_data.SurgeryLength < 1000] # length in minutes? removed long surgery outliers

y = rambam_data.SurgeryLength.values
features = rambam_data.columns
print(features)


X = rambam_data.drop("SurgeryLength", axis=1)
X = X.drop("ID_NO",axis=1)
X = X.drop("ADMISSION_NO",axis=1)
#X = X.drop("OPR_DATE",axis=1)
#X = X.drop("START_TIME",axis=1)
#X = X.drop("END_TIME",axis=1)
#X = X.drop("Date_Enter",axis=1)
#X = X.drop("Time_Exit",axis=1)
#X = X.drop("Date_Exit",axis=1)
#X = X.drop("ID",axis=1)
#X = X.drop("Birth",axis=1)
#X = X.drop("Visit#",axis=1)
#X = X.drop("Time_Enter",axis=1)
X = X.drop("LOS[days]",axis=1)
#
#X = X.drop(["F1","F8"],axis=1)
#X = X.to_numpy()
# Extract text features
cats = X.select_dtypes(exclude=np.number).columns.tolist()
for col in cats:
   print("converting to cat: ", col)
   X[col] = X[col].astype('category')
X.columns = X.columns.astype(str)
#X = X.drop("F1",axis=1)
#X = X.drop(["F7","F8","F9","F10"],axis=1)
#X = X.drop("Departement",axis=1)


#X['Gender'].replace(['נקבה', 'זכר'],[0, 1], inplace=True)
#X['F6'].replace(['אשפוז','מיון'],[1,0], inplace=True)

#encoded = pd.get_dummies(X.Departement)
# Concatenate the dummies to original dataframe
#X = pd.concat([X, encoded], axis='columns')
# dropping the original column which was not encoded
#X = X.drop(['Departement'], axis='columns')



X = pd.get_dummies(X,drop_first=True)


features = X.columns
print(features)
X.head()
X.info()
#%X = X.to_numpy()
#rf = RuleFit(tree_size=20,max_rules=200,model_type='rl')
#rf.fit(X, y, feature_names=features)
#y_pred = rf.predict(X)

#insample_rmse = np.sqrt(np.sum((y_pred - y)**2)/len(y))
#print("In sample RMSE=", insample_rmse)
#rules = rf.get_rules() #(exclude_zero_coef=True)
#rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
#pd.set_option("display.max_colwidth", 150)
#rules.head(10)
#print(rules)
##alldata = xgb.DMatrix(X,y,enable_categorical=True)

params = {"objective": "reg:squarederror","max_depth":3}

n = 1

##  results = xgb.cv(params=params, dtrain=alldata, num_boost_round=n, nfold=5, metrics={"rmse"}) #evals=evals)
dt = DecisionTreeRegressor(max_depth=3)
##print(results)
dt_fit = dt.fit(X, y)

y_pred = dt.predict(X)


insample_rmse = np.sqrt(np.sum((y_pred - y)**2)/len(y))
print("In sample RMSE=", insample_rmse)

dt_scores = cross_val_score(dt_fit, X, y, cv = 5, scoring='neg_root_mean_squared_error')
print("mean cross validation score: {}".format(np.mean(dt_scores)))
#print("score without cv: {}".format(dt_fit.score(X_train, y_train)))


##test_rmse_mean = results['test-rmse-mean'].mean()
##train_rmse_mean = results['train-rmse-mean'].mean()

##print("train avg rmse: ", train_rmse_mean, " test avg remse: ", test_rmse_mean)

#preds = model.predict(dtest_reg)

# 0.25 default test size
##X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
##dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
##dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)
##evals = [(dtest_reg, "validation"), (dtrain_reg, "train")]
##model = xgb.train(
##   params=params,
##   dtrain=dtrain_reg,
##   num_boost_round=n,
##   evals=evals,
##   verbose_eval=50,
   # Activate early stopping
##   early_stopping_rounds=50
##)

#preds = model.predict(dtest_reg)
##rmse = mean_squared_error(y_test, preds, squared=False)
##print(f"RMSE on the test: {rmse:.3f}")
##print(model)

y_class = dt.apply(X)
leaf_num = max(y_class)+1
ahat = np.zeros(len(y_class))
a = y_pred
for i in range(leaf_num):
   idxs = np.where(y_class == i)
   ahat[idxs] = np.percentile(y[idxs],AHAT_PTILE)

X.assign(a=a)
X.assign(ahat=ahat)

X.to_csv("Dep13300with_a_ahat.csv")


fig, ax = plt.subplots(figsize=(20, 20))
tree.plot_tree(dt,feature_names=features)
plt.show()

##fig, ax = plt.subplots(figsize=(20, 20))
##xgb.plot_tree(model, num_trees=0, ax=ax)
##plt.show()
##fig, ax = plt.subplots(figsize=(20, 20))
##xgb.plot_tree(model, num_trees=1, ax=ax)
##plt.show()
##fig, ax = plt.subplots(figsize=(20, 20))
##xgb.plot_tree(model, num_trees=2, ax=ax)
##plt.show()