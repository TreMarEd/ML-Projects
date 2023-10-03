"""
This script solves dummy task 0 of the Advanced ML lecture at ETH. It is the same as in the intro ML course
such that I recyle my script from there. The task consists of training a predictor R^10 -> R on some 
provided training data. The solution is known to be the mean of the input data, such that a simple linear regression will 
solve the prediction problem trivially and exactly up to machine precision. Implemented a model selection workflow for 
sklearn-linear-models with feature transformation and feature rescaling: 13 different models are generated evaluated and  
the predictions of the best model are written out.
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures


class Model:
    """
    Represents a sklearn linear model including some name to be given to the model and feature transformations to be applied to the
    raw data. Used to implement the model selection workflow.

    INSTANCE VARIABLES:
    "name":             string describing the name given to the model, f.e. "polynomial regression of degree 6, lasso alpha=0.1"
    "sklearn_object":   model object from sklearn.linear_model, f.e. linear_regression, Ridge or Lasso
    "feature_trafo":    feature transformation object from sklearn.preprocessing, f.e. PolynomialFeatures or FunctionTransformer
    "cv_score":         float representing the cross validation score of the model, initialized to None
    """

    def __init__(self, name, sklearn_object, feature_trafo):
        self.name = name
        self.sklearn_object = sklearn_object
        self.feature_trafo = feature_trafo
        self.cv_score = None


def select_model(candidate_models, x, y, scoring_function="neg_mean_squared_error", cv=10):
    """
    Given a multitude of candidate models returns the name of the model with the best CV score and prints the scores. 

    ARGUMENTS:
    'candidate_models': List of model objects from which the best model will be chosen
    'x':                n x d training data matrix, with n observations, each of dimension d
    'y':                Training label vector of dimension n
    'scoring_function': name of the function to be used to define the cv score
    'cv':               CV batch size

    RETURNS: 
    The key (meaning the model name) of the model with the highest CV-score
    """

    for candidate in candidate_models:
        x_trafo = trafo_scale_features(x, candidate.feature_trafo)
        cv_result = cross_validate(candidate.sklearn_object, x_trafo, y, cv=10, scoring=scoring_function)
        candidate.cv_score = np.mean(cv_result["test_score"])

    # get model name of model with highest cv-score
    best_model = max(candidate_models, key=lambda x: x.cv_score)

    # sort according to scores
    candidate_models = sorted(candidate_models, key=lambda x: x.cv_score, reverse=True)

    print("\nThe models have the following CV-scores from best to worst:")
    for candidate in candidate_models:
        print("\n", candidate.name, ": ", candidate.cv_score)

    return best_model


def trafo_scale_features(x, feature_trafo):
    """
    Applies feature tranformations and feature min/max rescaling on some data set x.

    ARGUMENTS:
    'x':             n x d dataframe with n observations of dimension d
    'feature_trafo': an sklear.preprocessing feature transformation object, f.e. PolynomialFeatures, to be used for 
                     feature transformation of the raw data x. Needs to be None if no feature transformation should be applied

    RETURNS:
    dataframe with n observations containing the transformed and min/max rescaled data. Equal to input if feature_trafo is None
    """
    if feature_trafo is None:
        return x
    else:
        x_trafo = feature_trafo.fit_transform(x)
        scaler = MinMaxScaler()
        scaler.fit(x_trafo)
        x_trafo = scaler.transform(x_trafo)
        return x_trafo


if __name__ == "__main__":

    # IMPORT AND PREPARE TRAINING DATA
    train = pd.read_csv("./AdvancedML/task 0/Input/train.csv")
    # drop labels and IDs
    x_train = train.iloc[:, 2:]
    # drop features and IDs
    y_train = train.iloc[:, 1]

    # INITIALIZE ALL CANDIDATE MODELS
    # list of model objects from which the best model will be chosen
    candidate_models = []

    candidate_models.append(Model("Linear regression", linear_model.LinearRegression(), None))
    degree = 4
    # do not include bias as sklearn.linear_model by default calculates bias
    poly_trafo = PolynomialFeatures(degree, include_bias=False)

    for a in [0.001, 0.01, 0.1, 1, 10, 100]:
        tmp_ridge = Model("linear ridge regression " + str(a), linear_model.Ridge(a), None)
        candidate_models.append(tmp_ridge)

        tmp_lasso = Model("polynomial degree " + str(degree) + " lasso regression " + str(a), linear_model.Lasso(a), poly_trafo)
        candidate_models.append(tmp_lasso)

    # CROSS VALIDATION AND MODEL SELECTION
    best_model = select_model(candidate_models, x_train, y_train)

    # FULLY TRAIN SELECTED MODEL
    x_trafo = trafo_scale_features(x_train, best_model.feature_trafo)
    best_model.sklearn_object.fit(x_trafo, y_train)

    # MODEL PREDICTION AND RESULT OUTPUT
    # test data set without labels, predictions on this data set will be submitted for the project
    test = pd.read_csv("./AdvancedML/task 0/Input/test.csv")

    # submission template, to be overwritten with solution for hand-in
    sample = pd.read_csv("./AdvancedML/task 0/Input/sample.csv")

    x_predict = test.iloc[:, 1:]
    x_predict_trafo = trafo_scale_features(x_predict, best_model.feature_trafo)

    sample["y"] = best_model.sklearn_object.predict(x_predict_trafo)
    sample.to_csv("./AdvancedML/task 0/result.csv", index=False)
