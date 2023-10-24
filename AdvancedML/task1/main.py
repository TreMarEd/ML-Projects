"""

"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time



def data_preprocessing(alpha=1e-4, seed=6, restarts=2, max_iter=3, kernel=Matern(length_scale=0.3, nu=0.4, length_scale_bounds=(1e-12, 1000000.0))):
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
    X_test: matrix of floats: test input with features
    """

    # Load training features and labels
    X_train = pd.read_csv("./AdvancedML/task1/data/X_train.csv")
    y_train = pd.read_csv("./AdvancedML//task1/data/y_train.csv") 
    # Load test features
    X_test = pd.read_csv("./AdvancedML//task1/data/X_test.csv")

    X_train = X_train.drop('id', axis=1)
    y_train = y_train.drop('id', axis=1)
    X_test = X_test.drop('id', axis=1)

    # TODO: outlier detection
    #print("\nPERFORMING OUTLIER DETECTION\n")

    # data scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nTRAINING IMPUTER\n")
    # train imputer on both train and test features
    X_imp = np.vstack([X_train, X_test])

    #X_imp = X_train
    # only partial data for testing purposes
    #X_imp = X_imp[:70, :]
    #imp = IterativeImputer(estimator=GaussianProcessRegressor(kernel=RBF(length_scale=1.0, length_scale_bounds=(1e-08, 10000000.0)), alpha=alpha, n_restarts_optimizer=restarts, random_state=seed), max_iter=max_iter)
    #imp = IterativeImputer(estimator=GaussianProcessRegressor(kernel=RationalQuadratic(length_scale=1.0, alpha=1.5, length_scale_bounds=(1e-10, 10000000.0), alpha_bounds=(1e-10, 100000000.0)), alpha=alpha, n_restarts_optimizer=restarts, random_state=seed), max_iter=max_iter)
    #imp = IterativeImputer(estimator=GaussianProcessRegressor(kernel=DotProduct(sigma_0_bounds=(1e-08, 100000.0)), alpha=alpha, n_restarts_optimizer=restarts, random_state=seed), max_iter=max_iter)
    #imp = IterativeImputer(estimator=GaussianProcessRegressor(kernel=Matern(length_scale=0.3, nu=1.5, length_scale_bounds=(1e-12, 1000000.0)), alpha=alpha, n_restarts_optimizer=restarts, random_state=seed), max_iter=max_iter)
    imp = KNNImputer(n_neighbors=3, weights="uniform")
    imp.fit(X_imp)

    print("\nIMPUTING TRAIN AND TEST DATA\n")
    X_train_imp = imp.transform(X_train)
    X_test_imp = imp.transform(X_test)

    print("\nPERFORMING DIMENSIONALITY REDUCTION\n")
    pca = PCA(n_components=0.20, svd_solver="full")
    X_pca = np.vstack([X_train_imp, X_test_imp])
    pca.fit(X_pca)
    X_train_pca = pca.transform(X_train_imp)
    X_test_pca = pca.transform(X_test_imp)

    #plt.plot([x+1 for x in range(len(pca.explained_variance_ratio_))], pca.explained_variance_ratio_, 'r-')
    #plt.show()
    print("NEW FEATURE DIM:", np.shape(X_train_pca)[1])

    # GPR by default assumes gaussian prior with mean zero, the predictor prior hence predicts every patients age as 0 
    # and hence ascribes significantly probability to nonsensical negative values. Prior knowledge of patients age can
    # be incorporated by subtracting mean patient age from labels and then adding again after prediction
    y_prior = np.mean(y_train["y"])
    y_train = y_train["y"].subtract(y_prior).to_numpy()

    return X_train_pca, X_test_pca, y_train, y_prior


def modeling_and_prediction(X_train, X_test, y_train, y_prior, kernels):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, 
    y_train: array of float
    X_test: matrix of floats
    kernels: list of sklearn Kernel objects, kernels of the Gaussian Process Regressor to be compared through cross-validation

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    alpha = 1

    # initialize cv_score for each kernel, best kernel and its score
    cv_scores = pd.DataFrame(columns=["CV_score"], index=[str(kernel) for kernel in kernels])
    best_kernel = None
    best_score = -1000

    for kernel in kernels:
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=3, random_state=5)
        print("\nTRAINING THE FOLLOWING KERNEL", str(kernel), "\n")
        score = np.mean(cross_val_score(gpr, X_train, y_train, cv=10, scoring="r2"))
        cv_scores.loc[str(kernel), "CV_score"] = score

        if score > best_score:
            best_kernel = kernel
            best_score = score

    print("\nTHE KERNELS ACHIEVE THE FOLLOWING R2-CV-SCORES:\n")
    print(cv_scores)
    print("\nTHE BEST KERNEL IS ", str(best_kernel), ". IT WILL BE USED TO GENERATE THE FINAL RESULT\n")

    gpr = GaussianProcessRegressor(kernel=best_kernel, alpha=alpha, n_restarts_optimizer=3, random_state=5)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)

    # add average age estimate of prior to obtain final posterior estimate
    y_pred = y_pred + y_prior

    return y_pred


if __name__ == "__main__":

    X_train, X_test, y_train, y_prior = data_preprocessing()

    kernels = [RationalQuadratic(length_scale=1.0, alpha=1.5, length_scale_bounds=(1e-08, 1000000.0), alpha_bounds=(1e-08, 200000.0))
               #RBF(length_scale=1.0, length_scale_bounds=(1e-08, 10000000.0)),
               #RationalQuadratic(length_scale=1.0, alpha=1.5, length_scale_bounds=(1e-08, 1000000.0), alpha_bounds=(1e-05, 200000.0)) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-06, 200000.0))#,
               #RBF(length_scale=1.0, length_scale_bounds=(1e-08, 10000000.0)) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-06, 200000.0))
               ]

    y_pred = modeling_and_prediction(X_train, X_test, y_train, y_prior, kernels)

    # write final result to csv
    sample = pd.read_csv("./AdvancedML/task1/data/sample.csv")
    sample["y"] = y_pred
    sample.to_csv("./AdvancedML/task1/result.csv", header=True, index=False)

    print("\nRESULTS FILE SUCCESSFULLY GENERATED!")
