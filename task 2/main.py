"""
This task is the second project of the 2023 intro to ML course at ETH. It consists of using Gaussian Processes to predict Swiss 
(log-)energy prices based on the energy prices of other European countries and the season. The training data has high nullity
such that an imputation strategy needs to be implemented during preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


def data_preprocessing(alpha=1e-8, seed=5, restarts=3, max_iter=15, kernel=Matern(length_scale=0.3, nu=0.3, length_scale_bounds=(1e-08, 100000.0))):
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using the iterative imputer provided by sklearn using a Gaussian Process Regressor.

    Parameters
    ----------
    alpha: float, Value added to the diagonal of the kernel matrix during fitting of the Gaussian Process regressor
    seed: integer, determines random number generation used during Gaussian Process regressor fitting
    restarts: integer, number of restarts of the Gaussian Process optimizer for finding  kernel parameters which maximize log-marginal likelihood. 
    max_iter: integer, maximum number of integers used by the iterative imputer
    kernel: sklearn Kernel object, kernel of the Gaussian Process to be used for imputation

    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """

    # Load training data
    train = pd.read_csv("./task 2/train.csv")
    # Load test data
    test = pd.read_csv("./task 2/test.csv")

    # encode categorical variable "season" with integer values according to their natural order relation provided by temperature
    seasons = ['winter', 'spring', 'autumn', 'summer']
    temperatures = [1, 2, 3, 4]
    train = train.replace(to_replace=seasons, value=temperatures)
    test = test.replace(to_replace=seasons, value=temperatures)

    print("\nTRAINING TRAINING-DATA IMPUTER\n")
    imp_train = IterativeImputer(estimator=GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, random_state=seed), max_iter=max_iter)
    train_imputed = imp_train.fit_transform(train)
    X_train = np.delete(train_imputed, 2, axis=1)
    y_train = train_imputed[:, 2]

    print("\nTRAINING TEST-DATA IMPUTER\n")
    imp_test = IterativeImputer(estimator=GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, random_state=seed), max_iter=max_iter)
    # use both training data feautes and test data features to train the imputer
    all_features = pd.concat([train.drop("price_CHF", axis=1), test], ignore_index=True)
    imp_test.fit(all_features)
    X_test = imp_test.transform(test)

    return X_train, y_train, X_test


def modeling_and_prediction(X_train, y_train, X_test, kernels):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features
    kernels: list of sklearn Kernel objects, kernels of the Gaussian Process Regressor to be compared through cross-validation

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    # initialize cv_score for each kernel, best kernel and its score
    cv_scores = pd.DataFrame(columns=["CV_score"], index=[str(kernel) for kernel in kernels])
    best_kernel = None
    best_score = 0

    for kernel in kernels:
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, n_restarts_optimizer=3, random_state=5)
        score = np.mean(cross_val_score(gpr, X_train, y_train, cv=10, scoring="r2"))
        cv_scores.loc[str(kernel), "CV_score"] = score

        if score > best_score:
            best_kernel = kernel
            best_score = score

    print("\nTHE KERNELS ACHIEVE THE FOLLOWING R2-CV-SCORES:\n")
    print(cv_scores)
    print("\nTHE BEST KERNEL IS ", str(best_kernel), ". IT WILL BE USED TO GENERATE THE FINAL RESULT\n")

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, n_restarts_optimizer=3, random_state=5)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)

    return y_pred


if __name__ == "__main__":

    X_train, y_train, X_test = data_preprocessing()

    kernels = [DotProduct(),
               RBF(),
               Matern(length_scale=0.3, nu=0.3,
                      length_scale_bounds=(1e-08, 100000.0)),
               RationalQuadratic(length_scale=1.0, alpha=1.5, length_scale_bounds=(1e-08, 100000.0))]

    y_pred = modeling_and_prediction(X_train, y_train, X_test, kernels)

    # write final result to csv
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('./task 2/results.csv', index=False)
    print("\nRESULTS FILE SUCCESSFULLY GENERATED!")
