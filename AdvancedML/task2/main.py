"""
The following script solves task 2 of the Advanced Machine Learning 2023 course at ETH Zurich.
TODO: more detailed explanation
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score #F1 = f1_score(y_true, y_pred, average='micro')
from biosppy.signals.ecg import ecg #TODO: maybe try another package, see this post: https://www.samproell.io/posts/signal/ecg-library-comparison/
import matplotlib.pyplot as plt

def get_wave_duration(template, time, peak_index, tolerance=0):
    """
    TODO: write docstring

    Parameters
    ----------

    Returns
    ----------
   
    """
    template_ = template + tolerance
    root_candidates = [np.sign(template_[i] * template_[i+1]) for i in range(len(template_)-1)]

    enumerated_roots = list(filter(lambda x: x[1] == -1, enumerate(root_candidates)))
    root_indices = list(map(lambda x: x[0], enumerated_roots))

    right_success = True
    left_success = True

    try:
        right_root_index = min(filter(lambda x: x > peak_index, root_indices))
    except ValueError:
        right_success = False
    try:
        left_root_index = max(filter(lambda x: x < peak_index, root_indices))
    except ValueError:
        left_success = False
    
    if right_success and left_success:
        right_time = time[right_root_index]
        left_time = time[left_root_index]
        
    
    elif right_success and not left_success:
       right_time = time[right_root_index] 
       left_time = 2*time[peak_index] - right_time
    
    elif not right_success and  left_success:
       left_time = time[left_root_index] 
       right_time = 2*time[peak_index] - left_time
    
    else: raise(ValueError)

    return right_time, left_time

    

    #TODO: special case where left or right root is not found for P and T wave


def find_extreme(template, time, index, side, ext):
    """
    TODO: write docstring
    Finds the minimum or maximum value of an array to the left or right of a certain index and returns the index and the value of template and time at that index
    Parameters
    ----------

    Returns
    ----------
   
    """
    if side == "right":

        template_ = template[index:]
        time_ = time[index:]

        if ext == "max":
            out_index = max(enumerate(template_), key=lambda x: x[1])[0]
            # add original index to out_index as the output should refer to the original template input
            return out_index + index, template_[out_index], time_[out_index]
        
        elif ext == "min":
            out_index = min(enumerate(template_), key=lambda x: x[1])[0]
            # add original index to out_index as the output should refer to the original template input
            return out_index + index, template_[out_index], time_[out_index]
        
        else:
            raise ValueError

    elif side == "left":

        template_ = template[:index]
        time_ = time[:index]

        if ext == "max":
            out_index = max(enumerate(template_), key=lambda x: x[1])[0]
            return out_index, template_[out_index], time_[out_index]
        
        elif ext == "min":
            out_index = min(enumerate(template_), key=lambda x: x[1])[0]
            return out_index, template_[out_index], time_[out_index]
        
        else:
            raise ValueError
    
    else:
        raise ValueError
    
    

def get_PQRST(template, template_ts):
    """
    TODO: write docstring
    Parameters
    ----------

    Returns
    ----------
   
    """

    # find P,Q,R,S,T times and signals, see https://en.wikipedia.org/wiki/QRS_complex

    # R-peak is the largest signal in a well behaved template
    R_index, R_signal, R_time = find_extreme(template, template_ts, 0, "right", "max")

    # S-peak is the smallest signal to the right of the R-peak in a well behaved template
    S_index, S_signal, S_time = find_extreme(template, template_ts, R_index, "right", "min")

    # Q-peak is the smallest signal to the left of the R-peak in a well behaved template
    Q_index, Q_signal, Q_time = find_extreme(template, template_ts, R_index, "left", "min")

    # T-peak is the largest signal to the right of the S-peak in a well behaved template
    T_index, T_signal, T_time = find_extreme(template, template_ts, S_index, "right", "max")

    # P-peak is the largest signal to the left of the Q-peak in a well behaved template
    P_index, P_signal, P_time = find_extreme(template, template_ts, Q_index, "left", "max")

    R_right_time, R_left_time = get_wave_duration(template, template_ts, R_index)
    P_right_time, P_left_time = get_wave_duration(template, template_ts, P_index, -5)
    Q_right_time, Q_left_time = get_wave_duration(template, template_ts, Q_index, 10)
    S_right_time, S_left_time = get_wave_duration(template, template_ts, S_index, 10)
    T_right_time, T_left_time = get_wave_duration(template, template_ts, T_index, -5)
    

    # find the wave durations for P,Q,R,S,T

    
    plt.figure()
    plt.plot(template_ts, template, '-r')
    plt.plot([P_time, Q_time, R_time, S_time, T_time], [P_signal, Q_signal, R_signal, S_signal, T_signal],'xb')
    plt.plot([P_right_time, P_left_time], [0,0],'ok')
    plt.plot([Q_right_time, Q_left_time], [0,0],'ok')
    plt.plot([R_right_time, R_left_time], [0,0],'ok')
    plt.plot([S_right_time, S_left_time], [0,0],'ok')
    plt.plot([T_right_time, T_left_time], [0,0],'ok')
    plt.show()
    print("hoi")
    
    

vget_PQRST = np.vectorize(get_PQRST)    

def get_features(sample, discard_first = 2):
    """
    TODO: write docstring
    Parameters
    ----------

    Returns
    ----------
   
    """

    """
    explanation of ecg outputs:
    ts: time axis for the outputs "filtered" and "rpeaks"
    filtered: array containing the denoised ECG signal relative to the time given in "ts"
    rpeaks: indices in ts and filtered of the rpeaks
    templates_ts: time axis of the "templates" output
    templates: each PQRST complex in the input is isolated. each row is one individual heartbeat
    heartrate_ts: time axis for the output "heart_rate"
    heart_rate: heart rate of the patient over time
    here ts, filtered and rpeaks are not used.
    """
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg(signal=sample[~np.isnan(sample)], sampling_rate=300, show = False)

    # TODO: apply get_PQRST to all templates and get means and variances (or medians and MADs)

vget_features = np.vectorize(get_features)  

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
    #X_train = pd.read_csv("./AdvancedML/task2/data/X_train.csv")
    X_train = pd.read_csv("./AdvancedML/task2/data/X_train_small.csv")
    #X_train.iloc[:100,:].to_csv("./AdvancedML/task2/data/X_train_small.csv", header=True, index=False)
    y_train = pd.read_csv("./AdvancedML//task2/data/y_train.csv")
    # Load test features
    #X_test = pd.read_csv("./AdvancedML//task2/data/X_test.csv")

    X_train = X_train.drop('id', axis=1).to_numpy()
    y_train = y_train.drop('id', axis=1).to_numpy()
    #X_test = X_test.drop('id', axis=1).to_numpy()

    print("\nEXTRACTING FEATURES\n")
    # TODO: find package that extracts PQRST complex for you. Then extract the mean and variances of PQRST times and signals, 
    # extract mean and variances of PQRST wave durations, QRS duration, total duration, venticular activation time, PR interval, QT interval
    # and differences of all signal pairs except maybe for P,Q relative to QRS. expect roughly 50 features
    #sample = X_train[14,:]
    #ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg(signal=sample[~np.isnan(sample)], sampling_rate=300, show = False)
    #get_PQRST(templates[10], templates_ts)



    sample = X_train[2,:]
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg(signal=sample[~np.isnan(sample)], sampling_rate=300, show = False)
    get_PQRST(templates[7], templates_ts)

    sample = X_train[3,:]
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg(signal=sample[~np.isnan(sample)], sampling_rate=300, show = False)
    get_PQRST(templates[7], templates_ts)

    sample = X_train[5,:]
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg(signal=sample[~np.isnan(sample)], sampling_rate=300, show = False)
    get_PQRST(templates[7], templates_ts)

    sample = X_train[44,:]
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg(signal=sample[~np.isnan(sample)], sampling_rate=300, show = False)
    get_PQRST(templates[7], templates_ts)

    sample = X_train[66,:]
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg(signal=sample[~np.isnan(sample)], sampling_rate=300, show = False)
    get_PQRST(templates[7], templates_ts)

    sample = X_train[77,:]
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg(signal=sample[~np.isnan(sample)], sampling_rate=300, show = False)
    get_PQRST(templates[7], templates_ts)


    # TODO: apply get_features to all samples

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
