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



def data_loading(random_state=0, max_iter=10):
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
    imp = IterativeImputer(max_iter=max_iter, random_state=random_state)
    imp.fit(train_imp)

    # transform regression features
    X_train = imp.transform(X_train_orig)
    X_test = imp.transform(X_test_orig)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred


if __name__ == "__main__":

    # use multiple imputation: m imputations with different RNG seeds
    m = 1

    # rational quadratic performs best
    kernels = [DotProduct(sigma_0=0.1), DotProduct(sigma_0=1), DotProduct(sigma_0=10), 
               RBF(length_scale=0.01), RBF(length_scale=1), RBF(length_scale=10), 
               Matern(length_scale=0.01), Matern(length_scale=0.1), Matern(length_scale=1), 
               RationalQuadratic(length_scale=0.1, alpha=10), RationalQuadratic(length_scale=10, alpha =0.1), RationalQuadratic(length_scale=1, alpha=1)]
    
    cv_scores = pd.DataFrame(index=[i for i in range(m)], columns=[str(k) for k in kernels])

    for random_state in range(m):
        for kernel in kernels:
            print("\n using kernel ", str(kernel), "\n")
            X_train, y_train, X_test = data_loading(random_state=random_state)

            gpr = GaussianProcessRegressor(kernel=kernel, alpha = 5e-8)
            #gpr.fit(X_train, y_train)
            cv_scores.loc[random_state, str(kernel)] = np.mean(cross_val_score(gpr, X_train, y_train, cv=10, scoring="r2"))

    print(np.array(cv_scores))

    # The function retrieving optimal LR parameters
    y_pred = modeling_and_prediction(X_train, y_train, X_test)

    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

