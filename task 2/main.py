'''
TODO: describe the task
'''

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

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

    # TODO: visually check the correctness of all these transforamtions
    # Load training data
    train_df = pd.read_csv("./task 2/train.csv")
    
    # Load test data
    test_df = pd.read_csv("./task 2/test.csv")

    # replace categorical variable season with a numeric value. The seasons have a natural order relation according to their average
    # temperatures in Europe. Energy prices are expected to be anticorrelated with the temperature of the season. Replace each season
    # with its average temperature in Kelvin according to the following page: 
    # https://weatherspark.com/y/46917/Average-Weather-in-Eu-France-Year-Round

    seasons = ['spring', 'summer', 'autumn', 'winter']
    temperatures = [282.0, 289.8, 285.4, 278.2] 

    train_df = train_df.replace(to_replace=seasons, value=temperatures)
    #original, non-imputed test data matrix 
    X_test_orig = test_df.replace(to_replace=seasons, value=temperatures)

    # visual inspection of the data shows that no column has significantly different nullity compared to the others except
    # for season, which is always present. Hence, no column should be dropped completely. Visual inspection also shows that 
    # there are almost no records with complete information, such that discarding missing rows is not feasible

    # drop records with missing labels for regression training
    train_reg = train_df.dropna(subset=['price_CHF'])
    y_train = train_reg['price_CHF']
    # original, non-imputed training data matrix
    X_train_orig = train_reg.drop(['price_CHF'], axis=1)

    # Use entire feature data set for multivariate imputation training
    train_imp = train_df.drop(['price_CHF'], axis=1)
    imp = IterativeImputer(max_iter=10)
    imp.fit(train_imp)

    # transform regression features
    X_train = imp.transform(X_train_orig)
    X_test = imp.transform(X_test_orig)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


if __name__ == "__main__":

    X_train, y_train, X_test = data_loading()

    kernels = [DotProduct(sigma_0=0.1),
               RBF(length_scale=0.1, length_scale_bounds=(1e-06, 100000.0)), 
               Matern(length_scale=0.2, nu=0.2), Matern(length_scale=0.3, nu=0.3), Matern(length_scale=0.4, nu=0.4), Matern(length_scale=0.5, nu=0.5),
               RationalQuadratic(length_scale=1, alpha=1)]
    
    cv_scores = pd.DataFrame(columns=["CV_score"], index=[str(k) for k in kernels])

    for kernel in kernels:
        print("\nUsing kernel ", str(kernel), "\n")
        gpr = GaussianProcessRegressor(kernel=kernel, alpha = 5e-9)
        cv_scores.loc[str(kernel), "CV_score"] = np.mean(cross_val_score(gpr, X_train, y_train, cv=10, scoring="r2"))

    # dot product always with bad performance, RBF with terrbible performance for all parametes,
    # matern and rational quadratic are competeteive, polynomial does not exist in the module. 
    # Choose matern(0.3, 0.3) in the following
    # TODO: try out more parameters for rational quadratic
    print(cv_scores)

    gpr = GaussianProcessRegressor(kernel=Matern(length_scale=0.3, nu=0.3), alpha = 5e-9)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)

    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('./task 2/results.csv', index=False)

    print("\nResults file successfully generated!")

