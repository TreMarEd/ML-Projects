"""
The following script solves task 1 of the Advanced Machine Learning 2023 course at ETH Zurich with an R^2 of 65%.

The project is a regression task where fMRI features are used to predict the biological age of a healthy person. The difference 
between the biological and the actual chronological age is a phenotype that in turn is useful in predicting the presence of a 
neurological desease.

The input data contains outliers, missing values and features must be selected on which to perfrom the regression.
I use an isolation forest for the outliers, the median imputer for the missing values and canonical correlation analysis (CCA)
for dimensionality reduction. Finally, after validating multiple kernels a Gaussian Process with Rational Quadratic kernel 
is used for regression.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import IsolationForest


def data_preprocessing(num_features):
    """
    Preprocesses and returns the data in the following way. Loads and imputes the data with the median. 
    Then an isolation forest is run to reject outliers. After standard scaling canonical correlation analysis (CCA) is run to
    extract num_features many features. 

    GPR by default assumes gaussian prior with mean zero, the predictor prior thus predicts every patients age as 0
    and hence ascribes significant probability to nonsensical negative values. Prior knowledge of patients age can
    be incorporated by subtracting mean patient age y_prior from labels and then adding again after prediction.

    Parameters
    ----------
    num_features: integer, number of features to be kept by CCA

    Returns
    ----------
    X_train: matrix of floats, preprocessed training features
    X_test: matrix of floats, preprocessed test features
    y_train: array of integers, preprocessed training labels with mean age subtracted
    y_prior: float, mean age of patients
    """

    # Load training features and labels
    X_train = pd.read_csv("./AdvancedML/task1/data/X_train.csv")
    y_train = pd.read_csv("./AdvancedML//task1/data/y_train.csv")
    # Load test features
    X_test = pd.read_csv("./AdvancedML//task1/data/X_test.csv")

    X_train = X_train.drop('id', axis=1)
    y_train = y_train.drop('id', axis=1)
    X_test = X_test.drop('id', axis=1)

    # GPR by default assumes gaussian prior with mean zero, the predictor prior hence predicts every patients age as 0
    # and hence ascribes significantly probability to nonsensical negative values. Prior knowledge of patients age can
    # be incorporated by subtracting mean patient age from labels and then adding again after prediction
    y_prior = np.mean(y_train["y"])
    y_train = y_train["y"].subtract(y_prior).to_numpy()

    print("\nIMPUTING DATA\n")
    imp = SimpleImputer(strategy='median')
    imp.fit(np.vstack([X_train, X_test]))
    X_train_imp = imp.transform(X_train.to_numpy())
    X_test_imp = imp.transform(X_test.to_numpy())

    print("\nPERFORMING OUTLIER DETECTION\n")
    isofor = IsolationForest(random_state=0)
    pred = isofor.fit_predict(X_train_imp)
    X_train_imp = X_train_imp[pred == 1]
    y_train = y_train[pred == 1]

    scaler = StandardScaler()
    scaler.fit(np.vstack([X_train_imp, X_test_imp]))
    X_train_scaled = scaler.transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    print("\nPERFORMING DIMENSIONALITY REDUCTION\n")
    # CCA on X and Y provides a loading vector for X which describes how much a given featrues correlates with y
    # for feature selection only keep the features with largest absolute loading value
    cca = CCA(n_components=1)
    cca.fit(X_train_scaled, y_train)
    abs_loadings = np.abs(cca.x_loadings_)
    # find the indices of the features with the largest absolute loading value
    indices = sorted(range(len(abs_loadings)), key=lambda x: abs_loadings[x])[-num_features:]
    X_train_cca = X_train_scaled[:, indices]
    X_test_cca = X_test_scaled[:, indices]

    return X_train_cca, X_test_cca, y_train, y_prior


def modeling_and_prediction(X_train, X_test, y_train, y_prior, kernels, alpha):
    """
    Given training and test data the function cross validates a list of Gaussian Processes with different kernels on the 
    training data, picks the best kernel, trains it on the full data and returns its prediction on the test data.

    Parameters
    ----------
    X_train: matrix of floats, preprocessed training features
    X_test: matrix of floats, preprocessed test features
    y_train: array of float, preprocesses training labels
    y_prior: float, mean age of patients in the training data, used as prior mean age 
    kernels: list of sklearn Kernel objects, kernels of the Gaussian Process Regressor to be compared through cross-validation
    alpha: float, equivalent to regularization parameter for ridge regression, term to be added to diagonal of Kernel matrix,
           interpreted as noise on the response variable

    Returns
    ----------
    y_test: array of floats: final predictions on X_test
    """

    # initialize cv_score for each kernel, best kernel and its score
    cv_scores = pd.DataFrame(columns=["CV_score"], index=[str(kernel) for kernel in kernels])
    best_kernel = None
    best_score = -100000

    for kernel in kernels:
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=3, random_state=58)
        print("\nTRAINING THE FOLLOWING KERNEL", str(kernel), "\n")
        score = np.mean(cross_val_score(gpr, X_train, y_train, cv=10, scoring="r2"))
        cv_scores.loc[str(kernel), "CV_score"] = score

        if score > best_score:
            best_kernel = kernel
            best_score = score

    print("\nTHE KERNELS ACHIEVE THE FOLLOWING R2-CV-SCORES:\n")
    print(cv_scores)

    print("\nTHE BEST KERNEL IS ", str(best_kernel),". IT WILL BE USED TO GENERATE THE FINAL RESULT\n")
    gpr = GaussianProcessRegressor(kernel=best_kernel, alpha=alpha, n_restarts_optimizer=3, random_state=5)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)

    # add average age estimate of prior to obtain final posterior estimate
    y_pred = y_pred + y_prior

    return y_pred


if __name__ == "__main__":

    X_train, X_test, y_train, y_prior = data_preprocessing(num_features=190)

    # list of kernels to be validated for the Gaussian process
    kernels = [RationalQuadratic(length_scale=1.0, alpha=1.5, length_scale_bounds=(1e-08, 1000000.0), alpha_bounds=(1e-08, 200000.0)),
               Matern(length_scale=0.3, nu=0.2, length_scale_bounds=(1e-12, 1000000.0)),
               RBF(),
               DotProduct()]

    y_pred = modeling_and_prediction(X_train, X_test, y_train, y_prior, kernels, alpha=1e-6)

    # write final result to csv using the provided sample submission
    sample = pd.read_csv("./AdvancedML/task1/data/sample.csv")
    sample["y"] = y_pred
    sample.to_csv("./AdvancedML/task1/result.csv", header=True, index=False)
    print("\nRESULTS FILE SUCCESSFULLY GENERATED!")
