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
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# define free parameters of the Gaussian Process


if __name__ == "__main__":

    # TODO: derive motivation for best_kernel being Matern

    # Load training data
    train = pd.read_csv("./task 2/train.csv")

    # Load test data
    test = pd.read_csv("./task 2/test.csv")

    # encode categorical variable "season" with integer values according to their natural order relation provided by temperature
    seasons = ['winter', 'spring', 'autumn', 'summer']
    temperatures = [1, 2, 3, 4] 

    train = train.replace(to_replace=seasons, value=temperatures)
    test = test.replace(to_replace=seasons, value=temperatures)

    best_kernel = Matern(length_scale=0.3, nu=0.3)
    alpha = 1e-5
    seed = 3
    restarts = 2
    
    imp_train = IterativeImputer(estimator=GaussianProcessRegressor(kernel=best_kernel, alpha=alpha, n_restarts_optimizer=restarts, random_state=seed), max_iter=10)
    imp_test = IterativeImputer(estimator=GaussianProcessRegressor(kernel=best_kernel, alpha=alpha, n_restarts_optimizer=restarts, random_state=seed), max_iter=10)
    
    print("\ntraining training-data imputer\n") 
    train_imputed = imp_train.fit_transform(train)
    print("\ntraining test-data imputer\n")
    test_imputed = imp_train.fit_transform(test)
   
    X_train = np.delete(train_imputed,2, axis=1)
    y_train = train_imputed[:,2]

    gpr = GaussianProcessRegressor(kernel=best_kernel, alpha=alpha, n_restarts_optimizer=restarts, random_state=seed)
    print("\n cross-validating predictor\n")
    score = np.mean(cross_val_score(gpr, X_train, y_train, cv=10, scoring="r2"))
    print("\nR2-CV-scores:\n")
    print(score)

    print("\n training final predictor\n")
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(test_imputed)

    # Save results in the required format
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('./task 2/results.csv', index=False)
    print("\nResults file successfully generated!")
