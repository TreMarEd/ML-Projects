'''
This task is the second project of the 2023 intro to ML course at ETH. It consists of using Gaussian Processes to predict Swiss 
(log-)energy prices based on the energy prices of other European countries and the season. The training data has high nullity
such that an imputation strategy needs to be implemented during preprocessing.
'''

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, Exponentiation
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# TODO: general cosmetics (do this at the end)

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """

    # Load training data
    train_0 = pd.read_csv("./task 2/train.csv")

    # Load test data
    test_0 = pd.read_csv("./task 2/test.csv")

    # replace categorical variable season with a numeric value. The seasons have a natural order relation according to their average
    # temperatures in Europe. Replace each season with its average temperature in Kelvin/100 according to the following page:
    # https://weatherspark.com/y/46917/Average-Weather-in-Eu-France-Year-Round
    seasons = ['spring', 'summer', 'autumn', 'winter']
    temperatures = [282.0, 289.8, 285.4, 278.2]

    train = train_0.replace(to_replace=seasons, value=temperatures)
    # non-imputed, i.e. original, test data matrix
    X_test_orig = test_0.replace(to_replace=seasons, value=temperatures)

    # drop records with missing labels for regression training
    train_reg = train.dropna(subset=['price_CHF'])
    y_train = train_reg['price_CHF']
    # non-imputed, i.e. original, training data matrix
    X_train_orig = train_reg.drop(['price_CHF'], axis=1)

    # Use entire feature data set for multivariate imputation training
    train_imp = train.drop(['price_CHF'], axis=1)
    imp = IterativeImputer(max_iter=20)
    imp.fit(train_imp)

    # transform regression features
    X_train = imp.transform(X_train_orig)
    X_test = imp.transform(X_test_orig)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


if __name__ == "__main__":

    X_train, y_train, X_test = data_loading()

    # TODO: try out composite kernels
    kernels = [DotProduct(sigma_0=0.1),
               RBF(length_scale=0.1, length_scale_bounds=(1e-06, 100000.0)),
               Matern(length_scale=0.3, nu=0.3),
               RationalQuadratic(length_scale=1, alpha=1),
               Exponentiation(RationalQuadratic(length_scale=0.1), exponent=2),
               Exponentiation(Matern(length_scale=1, nu=1), exponent=2),
               Exponentiation(RationalQuadratic(length_scale=0.1), exponent=3),
               Exponentiation(Matern(length_scale=1, nu=1), exponent=3)]
    
    # define open parameters of the Gaussian Process
    alpha = 5e-8
    random_state = 5
    n_restarts_optimizer = 5

    # initialize dataframe that holds cv score for each tried out kernel
    cv_scores = pd.DataFrame(columns=["CV_score"], index=[str(k) for k in kernels])

    # initialize best score and best kernel
    best_score = 0
    best_kernel = None 

    for kernel in kernels:
        print("\nUsing kernel ", str(kernel), "\n")
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer, random_state=random_state)

        score = np.mean(cross_val_score(gpr, X_train, y_train, cv=10, scoring="r2"))
        cv_scores.loc[str(kernel), "CV_score"] = score

        if score > best_score:
            best_score = score
            best_kernel = kernel

    print("\nThe kernels have the following R2-CV-scores:\n")
    print(cv_scores)
    print("\nBest kernel is ", str(best_kernel), " and it will be used for training and generating the results.")

    gpr = GaussianProcessRegressor(kernel=best_kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer, random_state=random_state)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)

    # Save results in the required format
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('./task 2/results.csv', index=False)
    print("\nResults file successfully generated!")
