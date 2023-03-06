"""
This script solves task 0 of the Intro to ML lecture at ETH.
The task consists of training a predictor R^10 -> R on some provided training data.
The solution is known to be the mean of the input data.
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate

def select_model(candidate_models, x, y, scoring_function="neg_root_mean_squared_error", cv=10):
    """
    Given a dictionary of candidate models returns the name of the model with the best CV score and prints the scores. 

    'candidate_models': dictionary, key is a string containing the model name, values is a 2-tuple containing the abstract model object
                        and the function used for feature transformation. The latter is None if no feature transformation is applied
    'x':                n x d training data matrix, with n observations, each of dimension d.
    'y':                Training label vector of dimension n
    'scoring_function': name of the function to be used to define the cv score
    'cv':               CV batch size

    returns: The key (meaning the model name) of the model with the highest CV-score
    """

    cv_scores = {}

    for model_name, model_def in candidate_models.items():

        model = model_def[0]
        feature_trafo = model_def[1]

        if feature_trafo is None:
            x_trafo = x
        else:
            raise Exception("feature trafos are not yet implemented!!!")
            
        # note: all scorer objects follow the convention that higher return values are better than lower return values
        cv_result = cross_validate(model, x_trafo, y, cv=10, scoring=scoring_function)    
        cv_scores[model_name] = np.mean(cv_result["test_score"])
    
    # get model name of model with highest cv-score
    best_model_name = max(cv_scores, key=cv_scores.get)

    print("\nThe models have the following CV-scores:")

    for model_name, score in cv_scores.items():
        print("\n", model_name, ": ", score)

    print("\nThe best model is:\n", best_model_name)

    return best_model_name


if __name__ == "__main__":

    # IMPORT AND PREPARE TRAINING DATA
    train = pd.read_csv("./task 0/Input/train.csv")
    x_train = train.iloc[:, 2:]
    y_train = train.iloc[:, 1]

    # INITIALIZE ALL CANDIDATE MODELS

    # candidate_models is a dictionary which maps a model name to a tuple with the initialized model and the feature transformation 
    # function to be applied to the raw data. If the second entry is None, no feature transformation is applied.
    # At some point this might have to be extended for the choice of optimization algorithm, but right now optimization is analytical
    candidate_models = {}

    candidate_models["linear regression"] = (linear_model.LinearRegression(), None)

    for alpha in [0.01, 0.1, 1, 10, 100]:
        candidate_models["linear ridge regression " + str(alpha)] = (linear_model.Ridge(alpha), None)

    # CROSS VALIDATION AND MODEL SELECTION

    best_model_name = select_model(candidate_models, x_train, y_train)
    best_model = candidate_models[best_model_name][0]
    feature_trafo = candidate_models[best_model_name][1] 

    # FULLY TRAIN SELECTED MODEL

    if feature_trafo is None:
        best_model.fit(x_train, y_train)
    else:
        raise Exception("feature trafos are not yet implemented here!!")

    # MODEL PREDICTION AND RESULT OUTPUT

    # test data set without labels, predictions on this data set will be submitted for the project
    test = pd.read_csv("./task 0/Input/test.csv")

    # submission template, to be overwritten with my solution for hand-in
    sample = pd.read_csv("./task 0/Input/sample.csv")
    sample["y"] = best_model.predict(test.iloc[:, 1:])
    sample.to_csv("./task 0/result.csv", index=False)
