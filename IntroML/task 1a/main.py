"""
This script solves task 1a of the Intro to ML lecture at ETH. The task consists of applying 10-fold cross-validation on multiple 
ridge-regression models defined by their regularization parameters for a predicton problem R^13 -> R on some provided training data. 
The solution consists of the cross-validation losses for each ridge-regression model.
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

train = pd.read_csv("./task 1a/train.csv")
# extract features
x_train = train.iloc[:, 1:]
# extract labels
y_train = train.iloc[:, 0]

RMSEs = []

for l in [0.1, 1., 10, 100, 200]:
    # "Ridge" by default chooses a solver by itself. Some solvers are stochastic and hence not necessarily reproducible on 
    # another machine. Hence, the cholesky solver is used as its solution is determinsitic
    ridge_model = linear_model.Ridge(alpha=l, solver="cholesky")
    cv_scores = cross_val_score(ridge_model, x_train, y_train, cv=10, scoring="neg_root_mean_squared_error")
    # by def. the higher the cv_score the better. Hence, negative RMSE is used and to get RMSE one needs to multiply by -1
    RMSE = -np.mean(cv_scores)
    RMSEs.append(RMSE)

# submission template, to be overwritten with solution for hand-in
sample = pd.read_csv("./task 1a/sample.csv", header=None)
sample.iloc[:, 0] = RMSEs
sample.to_csv("./task 1a/result.csv", header=False, index=False)
