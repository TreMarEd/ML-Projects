"""
This script solves task 0 of the Intro to ML lecture at ETH.
The task consists of training a predictor R^10 -> R on some provided training data.
The solution is known to be the mean of the input data.
"""

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

# MODEL TRAINING

train = pd.read_csv("./task 0/Input/train.csv")
x_train = train.iloc[:, 2:]
y_train = train.iloc[:, 1]

lin_reg = linear_model.LinearRegression()
lin_reg.fit(train.iloc[:, 2:], train.iloc[:, 1])

# As the correct predictor is the mean function the coefficients should all be 0.1 with intercept 0
print("\nregression coefficients: \n", lin_reg.coef_)
print("\nintercept \n", lin_reg.intercept_)

rmse = mean_squared_error(y_train, lin_reg.predict(x_train))**0.5

# MODEL PREDICTION AND OUTPUT

# test data set without labels, predictions on this data set will be submitted for the project
test = pd.read_csv("./task 0/Input/test.csv")

# submission template, to be overwritten with my solution
sample = pd.read_csv("./task 0/Input/sample.csv")
sample["y"] = lin_reg.predict(test.iloc[:, 1:])
sample.to_csv("./task 0/result.csv", index=False)

