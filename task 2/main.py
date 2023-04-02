'''
This task is the second project of the 2023 intro to ML course at ETH. It consists of using Gaussian Processes to predict Swiss 
(log-)energy prices based on the energy prices of other European countries and the season. The training data has high nullity
such that an imputation strategy needs to be implemented during preprocessing.
'''
# TODO: cosmetics

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, Exponentiation
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def impute_and_cv(train, test, kernel, alpha=1e-8, seed=4, restarts=1, max_iter=5):
    """
    TODO: fkt beschreiben
    """
    # TODO: eigener optimizer für gaussian process schreiben
    print("\nIMPUTATION USING KERNEL ", str(kernel), "\n")
    imp_train = IterativeImputer(estimator=GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, random_state=seed), max_iter=max_iter)
    
    print("\nTRAINING TRAINING-DATA IMPUTER\n") 
    train_imputed = imp_train.fit_transform(train)
    X_train = np.delete(train_imputed,2, axis=1)
    y_train = train_imputed[:,2]

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, random_state=seed)
    print("\nCROSS-VALIDATING PREDICOTR\n")
    score = np.mean(cross_val_score(gpr, X_train, y_train, cv=10, scoring="r2"))

    print("\nTRAINING PREDICTOR\n")
    gpr.fit(X_train, y_train)

    print("\nTRAINING TEST-DATA IMPUTER\n")
    imp_test = IterativeImputer(estimator=GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, random_state=seed), max_iter=max_iter)
    test_imputed = imp_test.fit_transform(test)

    return gpr, score, test_imputed


if __name__ == "__main__":

    # Load training data
    train = pd.read_csv("./task 2/train.csv")
    # Load test data
    test = pd.read_csv("./task 2/test.csv")

    # encode categorical variable "season" with integer values according to their natural order relation provided by temperature
    seasons = ['winter', 'spring', 'autumn', 'summer']
    temperatures = [1, 2, 3, 4]
    train_unscaled = train.replace(to_replace=seasons, value=temperatures)
    test_unscaled = test.replace(to_replace=seasons, value=temperatures)

    #TODO: scale the data, scaling needs to be inverted for final predictions
    scaler = StandardScaler()
    train = scaler.fit_transform(train_unscaled)

    test_unscaled["price_CHF"] = np.zeros(test_unscaled.shape[0])
    cols = train_unscaled.columns.to_list()
    test_unscaled = test_unscaled[cols]
    test = scaler.transform(test_unscaled)
    test = np.delete(test, 2, axis=1)

    #kernels = [Matern(length_scale=0.3, nu=0.3), 
               #RationalQuadratic(length_scale=1.0, alpha=1.5, length_scale_bounds=(1e-07, 100000.0))]
    kernels=[]

    # initialize cv_score for each kernel and best kernel and its score
    cv_scores = pd.DataFrame(columns=["CV_score"], index=[str(kernel) for kernel in kernels])
    best_kernel = None
    best_score = 0

    for kernel in kernels:
        score = impute_and_cv(train, test, kernel)[1]
        cv_scores.loc[str(kernel),"CV_score"] = score

        if score > best_score:
            best_kernel = kernel
            best_score = score
    
    print("\nTHE KERNELS ACHIEVE THE FOLLOWING R2-CV-SCORES:\n")
    print(cv_scores)
    print("\nTHE BEST KERNEL IS ", str(best_kernel), ".")
    print("\nIT WILL BE USED TO IMPUTE TRAINING/TEST DATA AGAIN WITH MORE ITERATIONS/OPTIMIZER RESTARTS AND TO GENERATE THE FINAL RESULT.\n")
    best_kernel = RationalQuadratic(length_scale=1.0, alpha=1.5, length_scale_bounds=(1e-07, 100000.0))
    best_kernel = Matern(length_scale=0.3, nu=0.3)
    gpr, score, test_imputed = impute_and_cv(train, test, best_kernel, restarts=4, max_iter=20)
    print("\nR2-CV-SCORE OF FINAL PREDICTOR: ", str(score), "\n")

    # generate final predictions and save to csv
    y_pred = gpr.predict(test_imputed)
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('./task 2/results.csv', index=False)
    print("\nRESULTS FILE SUCCESSFULLY GENERATED!")
