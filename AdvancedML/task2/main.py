"""
The following script solves task 2 of the Advanced Machine Learning 2023 course at ETH Zurich. The problem is a classification task:
Given raw heart ECG data the illness of the patient needs to predicted. The main problem of the task is the feature extraction from the raw
ECGs, see https://en.wikipedia.org/wiki/QRS_complex.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from biosppy.signals.ecg import ecg
import matplotlib.pyplot as plt
import time
from scipy import stats


def find_extremum(template, time, index, side, ext, a=-10000, b=10000):
    """
    
    Given an ECG signal as defined by template and time finds the minimum or maximum value of an array to the left or 
    right of a certain index and returns the index, the value of the template and time at that index. The search for the extremum
    can be constrained in the time domain using the parameters a and b.

    Parameters:
    template: 1d np array containing a PQRST complex from an ECG
    time: 1d np array containing the measurement times for the values given in template
    index: int, index relative to which the search for the extremum will take place
    side: "left" or "right", the side relative to "index" where the extremum will be searched for
    ext: "min" or "max", specifies whether to search for a minimum or maximum
    a: float, constrains the time of the extremum to be bigger than a
    b: float, constrains the time of the extremum to be smaller than b
    ----------
    Returns:
    out_index: int, index of the extremum
    out_signal: float, signal of the extremum
    out_time: float, time of the extremum
    """

    if side == "right":

        template_ = template[index:]
        time_ = time[index:]

        if ext == "max":
            out_index = max(filter(lambda x: time_[x[0]] < b and time_[x[0]] > a , enumerate(template_)), key=lambda x: x[1])[0]
            # add original index to out_index as the output should refer to the original template input
            return out_index + index, template_[out_index], time_[out_index]
        
        elif ext == "min":
            out_index = min(filter(lambda x: time_[x[0]] < b and time_[x[0]] > a , enumerate(template_)), key=lambda x: x[1])[0]
            # add original index to out_index as the output should refer to the original template input
            return out_index + index, template_[out_index], time_[out_index]
        
        else:
            raise ValueError

    elif side == "left":

        template_ = template[:index]
        time_ = time[:index]

        if ext == "max":
            out_index = max(filter(lambda x: time_[x[0]] < b and time_[x[0]] > a , enumerate(template_)), key=lambda x: x[1])[0]
            return out_index, template_[out_index], time_[out_index]
        
        elif ext == "min":
            out_index = min(filter(lambda x: time_[x[0]] < b and time_[x[0]] > a , enumerate(template_)), key=lambda x: x[1])[0]
            return out_index, template_[out_index], time_[out_index]
        
        else:
            raise ValueError
    else:
        raise ValueError
    

def get_features(sample):
    """
    Takes a single ECG sample as input and returns the extracted features. The features extracted are the P, Q, R, S, T signals and times,
    ratios/differences of these signals and times and standard quantities from signals processing like fourier modes and energy

    Parameters:
    sample: 1d array, containing the raw ECG data with a sampling rate of 300 HZ
    ----------
    Returns:
    output: dictionary, with (key, values) corresponding to feature names and values
    ----------
    """

    output = {}
    
    sample = sample[~np.isnan(sample)]
    
    output["avg"] = np.mean(sample)
    output["autocorr2"] = pd.Series(sample).autocorr(lag=2)
    output["autocorr6"] = pd.Series(sample).autocorr(lag=6)

    f = np.fft.fft(sample[:2400])
    sorted_modes = np.abs(f).argsort()
    for i in range(10):
        output["fft_modes_"+str(i)] = sorted_modes[-i-1]

    # https://biosppy.readthedocs.io/en/stable/biosppy.signals.html#biosppy-signals-ecg
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg(signal=sample, sampling_rate=300, show = False)

    mean_template = np.mean(templates, axis=0)

    output["max"] = np.max(mean_template)
    output["min"] = np.max(mean_template)
    output["mean"] = np.mean(mean_template)
    output["median"] = np.median(mean_template)
    output["energy"] = np.sum(mean_template**2)

    # R-peak is the largest signal in a well behaved template
    output["R_index"], output["R_signal"], R_time = find_extremum(mean_template, templates_ts, 0, "right", "max", a=-0.06, b=0.06 )
    # S-peak is the smallest signal to the right of the R-peak in a well behaved template
    output["S_index"], output["S_signal"], S_time = find_extremum(mean_template, templates_ts, output["R_index"], "right", "min", a=0, b=0.08)
    # Q-peak is the smallest signal to the left of the R-peak in a well behaved template
    output["Q_index"], output["Q_signal"], Q_time = find_extremum(mean_template, templates_ts, output["R_index"], "left", "min", a=-0.08, b=0)
    # T-peak is the largest signal to the right of the S-peak in a well behaved template
    output["T_index"], output["T_signal"], T_time = find_extremum(mean_template, templates_ts, output["S_index"], "right", "max", a=0.19)
    # P-peak is the largest signal to the left of the Q-peak in a well behaved template
    output["P_index"], output["P_signal"], P_time = find_extremum(mean_template, templates_ts, output["Q_index"], "left", "max", b=-0.08)

    output["PR_duration"] = output["R_index"] - output["P_index"]
    output["QS_duration"] = output["S_index"] - output["Q_index"]
    output["ST_duration"] = output["T_index"] - output["S_index"]

    # get ratios and differences of the signal pairs. P and T are avoided as their extraction is very volatile
    for i, n in enumerate(["Q", "R", "S"]):
        for j, m in enumerate(["Q", "R", "S"]):
            if i>=j: 
                continue
            output[n + "/" + m + "_signal"] = output[n + "_signal"] / output[m + "_signal"]
            output[n + "-" + m + "_signal"] = output[n + "_signal"] - output[m + "_signal"]

    diff_heart_rate_ts = np.diff(heart_rate_ts)
    output["heart_rate_ts_mean"] = np.mean(diff_heart_rate_ts)
    output["heart_rate_ts_std"] = np.std(diff_heart_rate_ts)
    output["heart_rate_ts_median"] = np.median(diff_heart_rate_ts)

    diff_rpeaks = np.diff(rpeaks)
    output["rpeaks_diff_mean"] = np.mean(diff_rpeaks)
    output["rpeaks_diff_median"] = np.median(diff_rpeaks)
    output["rpeaks_diff_std"] = np.std(diff_rpeaks)
    output["rpeaks_diff_mode"] = stats.mode(diff_rpeaks)[0]

    output["heart_rate_mean"] = np.mean(heart_rate)
    output["heart_rate_std"] = np.std(heart_rate)
    output["heart_rate_median"] = np.median(heart_rate)

    return output


def data_preprocessing():
    """
    Reads in the raw ECG data and extracts the relevant features.

    Parameters:
    None
    ----------
    Returns:
    features_train: 2d np array containing training features
    features_test: 2d np array containing test features
    y_train: 1d np array containing training labels
    """

    print("\nLOADING TRAINING DATA\n")
    X_train = pd.read_csv("./AdvancedML/task2/data/X_train.csv")
    y_train = pd.read_csv("./AdvancedML/task2/data/y_train.csv")

    print("\nLOADING TEST DATA\n")
    X_test = pd.read_csv("./AdvancedML/task2/data/X_test.csv")

    X_train = X_train.drop('id', axis=1).to_numpy()
    y_train = y_train.drop('id', axis=1).to_numpy().flatten()
    X_test = X_test.drop('id', axis=1).to_numpy()

    print("\nEXTRACTING TRAINING FEATURES\n")
    features_train = list(np.apply_along_axis(get_features, 1, X_train))
    features_train = pd.DataFrame(features_train)
    features_train = features_train.to_numpy()

    print("\nEXTRACTING TEST FEATURES\n") 
    features_test = list(np.apply_along_axis(get_features, 1, X_test))
    features_test = pd.DataFrame(features_test)
    features_test = features_test.to_numpy()

    print("\nIMPUTING DATA\n")
    # during feature extraction some means are taken from empty slices, the resulting nans are imputet here
    imp = SimpleImputer(strategy='median')
    imp.fit(np.vstack([features_train, features_test]))
    features_train = imp.transform(features_train)
    features_test = imp.transform(features_test)

    return features_train, features_test, y_train


def modeling_and_prediction(X_train, X_test, y_train, models):
    """
    Trains the provided Gradient Boosting Classification models on X_train and y_train, cross validates them, chooses the best one
    and returns the predictions based on X_test.

    Parameters:
    X_train: 2d np array containing training features
    X_test: 2d np array containing test features
    y_train: 1d np array containing training labels
    models: list of dictionaries containing Gradient Boosting Classification model specifications
    ----------
    Returns
    y_pred: 1d np arary containing the predictions w.r.t. X_test by the best model
    """

    # initialize cv_score for each model, best model and its score
    best_model = None
    best_score = -100000

    for i in range(np.shape(models)[0]):
        print("==========================================================") 
        model = models.iloc[i, :]
        print("\nTRAINING THE FOLLOWING MODEL\n", str(model), "\n")
        t0 = time.time()
        gbc = GradientBoostingClassifier(learning_rate=model["learning_rate"], n_estimators=int(model["n_estimators"]), max_depth=int(model["max_depth"]), 
                                         min_samples_split=int(model["min_samples_split"]), min_samples_leaf=int(model["min_samples_leaf"]), random_state=42)
        scores = cross_val_score(gbc, X_train, y_train, cv=10, scoring="f1_micro", error_score="raise")
        score = np.mean(scores)
        if score > best_score:
            best_model = model
            best_score = score
        print("\nMODEL GOT THE FOLLOWING SCORE", str(score), "\n")
        t1 = time.time() 
        print("\nELAPSED MINUTES:", (t1-t0)/60, "\n")

    print("\nTHE BEST MODEL IS ", str(best_model)," with a score of " + str(best_score)+". IT WILL BE USED TO GENERATE THE FINAL RESULT\n")
    gbc = GradientBoostingClassifier(learning_rate=best_model["learning_rate"], n_estimators=int(best_model["n_estimators"]), max_depth=int(best_model["max_depth"]), 
                                     min_samples_split=int(best_model["min_samples_split"]), min_samples_leaf=int(best_model["min_samples_leaf"]), random_state=42)
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    
    return y_pred


if __name__ == "__main__":

    t0 = time.time()
    X_train, X_test, y_train = data_preprocessing()
    t1 = time.time() 
    print("\nELAPSED MINUTES:", (t1 - t0) / 60, "\n")

    # define a dictionary containing a range of Gradient Boosting parameters to be cross validated
    models = []
    for n_estimators in [250]:
        for max_depth in [7]:
            for learning_rate in [0.05]:
                    for min_samples_leaf in [9]: 
                        for min_samples_split in [80]:              
                            model = {}
                            model["n_estimators"] = n_estimators
                            model["max_depth"] = max_depth
                            model["learning_rate"] = learning_rate
                            model["min_samples_leaf"] = min_samples_leaf
                            model["min_samples_split"] = min_samples_split
                            models.append(model)

    print("\nNUMBER OF MODELS TO BE TRAINED: ", len(models), "\n") 
    
    y_pred = modeling_and_prediction(X_train, X_test, y_train, pd.DataFrame(models))

    # write final result to csv using the provided sample submission
    sample = pd.read_csv("./AdvancedML/task2/data/sample.csv")
    sample["y"] = y_pred
    sample.to_csv("./AdvancedML/task2/result.csv", header=True, index=False)
    print("\nRESULTS FILE SUCCESSFULLY GENERATED!")
    