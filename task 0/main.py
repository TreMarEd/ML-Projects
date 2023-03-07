"""
This script solves task 0 of the Intro to ML lecture at ETH. The task consists of training a predictor R^10 -> R on some 
provided training data. The solution is known to be the mean of the input data, such that a simple linear regression will 
solve the prediction problem trivially and exactly. Implemented a model selection workflow for sklearn-linear-models with 
feature transformation and feature rescaling. The predictions of the best model are written out.
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler


def select_model(candidate_models, x, y, scoring_function="neg_root_mean_squared_error", cv=10):
    """
    Given a multitude of candidate models returns the name of the model with the best CV score and prints the scores. 

    ARGUMENTS:
    'candidate_models': type dictionary, key is a string containing the model name, values are 2-tuples containing the abstract 
                        sklearn-model-object and a list of the functios used for feature transformation. The latter is None 
                        if no feature transformation is applied.
    'x':                n x d training data matrix, with n observations, each of dimension d
    'y':                Training label vector of dimension n
    'scoring_function': name of the function to be used to define the cv score
    'cv':               CV batch size

    RETURNS: 
    The key (meaning the model name) of the model with the highest CV-score
    """

    # dictionary that maps model names to their cv-score
    cv_scores = {}

    for model_name, model_def in candidate_models.items():

        feature_trafo = model_def[1]
        model = model_def[0]

        x_trafo = trafo_scale_features(x, feature_trafo)

        cv_result = cross_validate(model, x_trafo, y, cv=10, scoring=scoring_function)
        cv_scores[model_name] = np.mean(cv_result["test_score"])

    # get model name of model with highest cv-score
    best_model_name = max(cv_scores, key=cv_scores.get)

    # sort according to scores
    cv_scores = {key: value for key, value in sorted(cv_scores.items(), key=lambda item: item[1], reverse=True)}

    print("\nThe models have the following CV-scores from best to worst:")
    for model_name, score in cv_scores.items():
        print("\n", model_name, ": ", score)

    return best_model_name


def trafo_scale_features(x, feature_trafo):
    """
    Applies feature tranformations and feature min/max rescaling on some data set x.

    ARGUMENTS:
    'x': n x d dataframe with n observations of dimension d
    'feature_trafo': a list of functions which perform the feature transformation.
                     f.e. if features are transformed with polynomials up to degree n, then the list holds 
                     the funtion x**1,x**2, ..., x**n. If None then x is returned

    RETURNS:
    dataframe with n observations containing the transformed and min/max rescaled data. Equal to input if feature_trafo is None
    """
    if feature_trafo is None:
        return x
    else:
        x_trafo = pd.concat([t(x_train) for t in feature_trafo], axis=1)
        scaler = MinMaxScaler()
        scaler.fit(x_trafo)
        x_trafo = scaler.transform(x_trafo)
        return x_trafo


if __name__ == "__main__":

    # IMPORT AND PREPARE TRAINING DATA

    train = pd.read_csv("./task 0/Input/train.csv")
    # drop labels and IDs
    x_train = train.iloc[:, 2:]
    # drop features and IDs
    y_train = train.iloc[:, 1]

    # INITIALIZE ALL CANDIDATE MODELS

    # candidate_models is a dict mapping a model name to a tuple with the initialized model and the feature transformation
    # function to be applied to the raw data. If the second entry is None, no feature transformation is applied.
    # Otherwise it is a list of functions, each of which will be applied to the features.
    candidate_models = {}

    candidate_models["linear regression"] = (linear_model.LinearRegression(), None)

    # list containg the feature transformations for polynomials up to degree 6
    # list comprehension is somewhat tricky with lambda functions, solution taken from here: 
    # https://stackoverflow.com/questions/6076270/lambda-function-in-list-comprehensions
    degree = 12
    polynomial_trafo = [(lambda y: (lambda x: x ** y))(i) for i in range(1, degree+1)]

    for a in [0.001, 0.01, 0.1, 1, 10, 100]:
        candidate_models["linear ridge regression " + str(a)] = (linear_model.Ridge(a), None)
        candidate_models["polynomial degree " + str(degree) + " lasso regression " + str(a)] = (linear_model.Lasso(a), polynomial_trafo)

    # CROSS VALIDATION AND MODEL SELECTION

    best_model_name = select_model(candidate_models, x_train, y_train)
    best_model = candidate_models[best_model_name][0]
    feature_trafo = candidate_models[best_model_name][1]

    # FULLY TRAIN SELECTED MODEL

    x_trafo = trafo_scale_features(x_train, feature_trafo)
    best_model.fit(x_trafo, y_train)

    # MODEL PREDICTION AND RESULT OUTPUT

    # test data set without labels, predictions on this data set will be submitted for the project
    test = pd.read_csv("./task 0/Input/test.csv")

    # submission template, to be overwritten with solution for hand-in
    sample = pd.read_csv("./task 0/Input/sample.csv")

    x_predict = test.iloc[:, 1:] 
    x_predict_trafo = trafo_scale_features(x_predict, feature_trafo)

    sample["y"] = best_model.predict(x_predict_trafo)
    sample.to_csv("./task 0/result.csv", index=False)
