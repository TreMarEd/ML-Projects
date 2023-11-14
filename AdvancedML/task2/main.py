"""
The following script solves task 2 of the Advanced Machine Learning 2023 course at ETH Zurich.
TODO: more detailed explanation
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
#F1 = f1_score(y_true, y_pred, average='micro')


def data_preprocessing():
    """
    TODO: write docstring
    Parameters
    ----------

    Returns
    ----------
   
    """

    # Load training features and labels
    print("\nLOADING DATA\n")
    X_train = pd.read_csv("./AdvancedML/task2/data/X_train.csv")
    y_train = pd.read_csv("./AdvancedML//task2/data/y_train.csv")
    # Load test features
    X_test = pd.read_csv("./AdvancedML//task2/data/X_test.csv")

    X_train = X_train.drop('id', axis=1)
    y_train = y_train.drop('id', axis=1)
    X_test = X_test.drop('id', axis=1)

    print("\nEXTRACTING FEATURES\n")
    # TODO: find package that extracts PQRST complex for you. Then extract the mean and variances of PQRST times and signals, 
    # extract mean and variances of PQRST wave durations, QRS duration, total duration, venticular activation time, PR interval, QT interval
    # and differences of all signal pairs except maybe for P,Q relative to QRS. expect roughly 50 features

    # TODO: remove nans

    # TODO: scale the data

    # TODO: dimred 

    # TODO: deal with class imbalance: either by resampling or using an SVM with class_weights = "balanced"

    return None, None, None, None


def modeling_and_prediction():
    """
    TODO: write docstring
    Parameters
    ----------

    Returns
    ----------

    """

    # TODO: choose model: SVM? bagged SVM? gradient boosting?

    # TODO: cross validate svm kernels and regularization parameter C

    # initialize cv_score for each kernel, best kernel and its score
    """
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
    """
    y_pred = None
    return y_pred
    


if __name__ == "__main__":

    X_train, X_test, y_train, y_prior = data_preprocessing()

    """
    # list of kernels to be validated for the Gaussian process
    kernels = [RationalQuadratic(length_scale=1.0, alpha=1.5, length_scale_bounds=(1e-08, 1000000.0), alpha_bounds=(1e-08, 200000.0)),
               Matern(length_scale=0.3, nu=0.2, length_scale_bounds=(1e-12, 1000000.0)),
               RBF(),
               DotProduct()]
    """
    y_pred = modeling_and_prediction()
    

    # write final result to csv using the provided sample submission
    sample = pd.read_csv("./AdvancedML/task2/data/sample.csv")
    sample["y"] = y_pred
    sample.to_csv("./AdvancedML/task2/result.csv", header=True, index=False)
    print("\nRESULTS FILE SUCCESSFULLY GENERATED!")
