"""
The following script solves task 2 of the Advanced Machine Learning 2023 course at ETH Zurich.
TODO: more detailed explanation
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

def get_wave_duration(template, time, peak_index):
    """
    TODO: write docstring

    Parameters
    ----------

    Returns
    ----------
   
    """
    template_ = template
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
        #linear interpolation between the two indices that define the root
        right_time = time[right_root_index + 1] - template[right_root_index + 1] * (time[right_root_index+1] - time[right_root_index])/(template[right_root_index+1]-template[right_root_index])
        left_time = time[left_root_index + 1] - template[left_root_index + 1] * (time[left_root_index+1] - time[left_root_index])/(template[left_root_index+1] - template[left_root_index])
    
    elif right_success and not left_success:
       #linear interpolation between the two indices that define the root
       right_time = time[right_root_index + 1] - template[right_root_index + 1] * (time[right_root_index+1] - time[right_root_index])/(template[right_root_index+1]-template[right_root_index])
       left_time = time[peak_index] - abs(right_time- time[peak_index])
    
    elif not right_success and  left_success:
       #linear interpolation between the two indices that define the root
       left_time = time[left_root_index + 1] - template[left_root_index + 1] * (time[left_root_index+1] - time[left_root_index])/(template[left_root_index+1] - template[left_root_index])
       right_time = time[peak_index] + abs(left_time - time[peak_index])
    
    else: 
        raise(ValueError)

    duration = right_time - left_time

    return right_time, left_time, duration


def find_extreme(template, time, index, side, ext, a=-10000, b=10000):
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
    

def get_PQRST(template, template_ts):
    """
    TODO: write docstring
    Parameters
    ----------

    Returns
    ----------
   
    """
    #if sample_index==4 and template_index==6:
    #    print("hoi")
    output = {}

    # find P,Q,R,S,T times and signals, see https://en.wikipedia.org/wiki/QRS_complex
    # R-peak is the largest signal in a well behaved template
    output["R_index"], output["R_signal"], output["R_time"] = find_extreme(template, template_ts, 0, "right", "max", a=-0.06, b=0.06 )

    # S-peak is the smallest signal to the right of the R-peak in a well behaved template
    output["S_index"], output["S_signal"], output["S_time"] = find_extreme(template, template_ts, output["R_index"], "right", "min", a=0, b=0.08)

    # Q-peak is the smallest signal to the left of the R-peak in a well behaved template
    output["Q_index"], output["Q_signal"], output["Q_time"] = find_extreme(template, template_ts, output["R_index"], "left", "min", a=-0.08, b=0)

    # T-peak is the largest signal to the right of the S-peak in a well behaved template
    output["T_index"], output["T_signal"], output["T_time"] = find_extreme(template, template_ts, output["S_index"], "right", "max", a=0.19)

    # P-peak is the largest signal to the left of the Q-peak in a well behaved template
    output["P_index"], output["P_signal"], output["P_time"] = find_extreme(template, template_ts, output["Q_index"], "left", "max", b=-0.08)

    #R_right_time, R_left_time, output["R_duration"] = get_wave_duration(template, template_ts, output["R_index"])
    #P_right_time, P_left_time, output["P_duration"] = get_wave_duration(template, template_ts, output["P_index"])
    #Q_right_time, Q_left_time, output["Q_duration"] = get_wave_duration(template, template_ts, output["Q_index"])
    #S_right_time, S_left_time, output["S_duration"] = get_wave_duration(template, template_ts, output["S_index"])
    #T_right_time, T_left_time, output["T_duration"] = get_wave_duration(template, template_ts, output["T_index"])

    # fill pairwise wave durations. it is unclear whether for example to go from start/end of P wave to start/end of T wave.
    # The medical literature has no systematic nomenclature, however all technical medical quantities are used below with their 
    # name in the literature (see comments). Otherwise, I apply the convention that for example the "PR duration" is the difference
    # between the start of R and the start of P

    #output["PR-interval"] = Q_left_time - P_right_time #technical medical term, called "PR" even though QP is concerned
    #output["PR-segment"] = Q_left_time - P_left_time #technical medical term called "PR" even though QP is concerned
    #output["PR-duration"] = R_left_time - P_left_time
    #output["PS-duration"] = S_left_time - P_left_time
    #output["total-duration"] = T_right_time - P_left_time #not a technical term, describes the total extent of the waveform
    #output["VAT"] = output["R_time"] - Q_left_time  #technical medical term: ventricular activation time
    #output["QRS-duration"] = S_right_time - Q_left_time #technical medical term
    #output["QT-duration"] = T_left_time - Q_left_time
    #output["ST-segment"] = T_left_time - S_right_time #technical medical term
    ##output["VDT"] = S_right_time - output["R_time"] # not a technical term but defined by me analogous to ventricular activation time: ventricular deactivation time
    #output["RT-duration"] = T_left_time - R_left_time
    #output["ST-interval"] = T_right_time - S_right_time #technical medical term

    return output
    

def get_features(sample):
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
    output = {}
    sample_wonan = sample[~np.isnan(sample)]
    sample_nan0 = np.nan_to_num(sample)
    output["autocorr"] = pd.Series(sample_wonan).autocorr(lag=2)
    output["avg"] = np.mean(sample_wonan)
    #output["ptp"] = np.ptp(sample_wonan) #kann der raus?

    f = np.fft.fft(sample_wonan[:2400])
    modes = np.abs(f).argsort()[-10:]
    for i in range(10):
        output["fft_modes_"+str(i)] = modes[i]

    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg(signal=sample_nan0, sampling_rate=300, show = False)

    mean_template = np.mean(templates, axis=0)
    output["energy"] = np.sum(mean_template**2)
    pqrst_result = get_PQRST(mean_template, templates_ts)

    output["P_index"] = pqrst_result["P_index"]
    output["Q_index"] = pqrst_result["Q_index"]
    output["R_index"] = pqrst_result["R_index"]
    output["S_index"] = pqrst_result["S_index"]
    output["T_index"] = pqrst_result["T_index"]

    output["PR_time"] = output["R_index"] - output["P_index"]
    output["QS_time"] = output["S_index"] - output["Q_index"]
    output["ST_time"] = output["T_index"] - output["S_index"]
    #output["QS_to_T_time"] = output["QS_time"]/(output["T_time"]+1) #kann der raus?
    #output["QS_to_P_time"] = output["QS_time"]/(output["P_time"]+1) #kann der raus?

    output["max"] = np.max(mean_template)
    output["min"] = np.max(mean_template)
    output["mean"] = np.mean(mean_template)
    output["median"] = np.median(mean_template)

    output["energy"] = pqrst_result["energy"]

    diff_heart_rate_ts = np.diff(heart_rate_ts)
    output["heart_rate_ts_mean"] = np.mean(diff_heart_rate_ts)
    output["heart_rate_ts_std"] = np.std(diff_heart_rate_ts)
    output["heart_rate_ts_median"] = np.median(diff_heart_rate_ts)

    diff_rpeaks = np.diff(rpeaks)
    output["peaks_diff_mean"] = np.mean(diff_rpeaks)
    output["peaks_diff_median"] = np.median(diff_rpeaks)
    output["peaks_diff_std"] = np.std(diff_rpeaks)
    output["peaks_diff_mode"] = stats.mode(diff_rpeaks)[0]

    output["heart_rate_mean"] = np.mean(heart_rate)
    output["heart_rate_std"] = np.std(heart_rate)
    output["heart_rate_median"] = np.median(heart_rate)

    output["P_signal"] = pqrst_result["P_signal"]
    output["Q_signal"] = pqrst_result["Q_signal"]
    output["R_signal"] = pqrst_result["R_signal"]
    output["S_signal"] = pqrst_result["S_signal"]
    output["T_signal"] = pqrst_result["T_signal"]

    for i, n in enumerate(["Q", "R", "S"]):
        for j, m in enumerate(["Q", "R", "S"]):
            if i>=j: 
                continue
            output[n + "/" + m + "_signal"] = pqrst_result[n + "_signal"]/pqrst_result[m + "_signal"]
            output[n + "-" + m + "_signal"] = pqrst_result[n + "_signal"]-pqrst_result[m + "_signal"]

    return output


def data_preprocessing():
    """
    TODO: write docstring
    Parameters
    ----------

    Returns
    ----------
   
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

    np.savetxt("./AdvancedML//task2/data/features_train_g3.csv", features_train, delimiter=",")
    np.savetxt("./AdvancedML//task2/data/features_test_g3.csv", features_test, delimiter=",")

    return features_train, features_test, y_train


def modeling_and_prediction(X_train, X_test, y_train, models):
    """
    TODO: write docstring
    Parameters
    ----------

    Returns
    ----------

    """

    # initialize cv_score for each kernel, best kernel and its score
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
        print("\nMODEL GOT THE FOLLOWING SCORE", str(score), "\n")
        t1 = time.time() 
        print("\nELAPSED MINUTES:", (t1-t0)/60, "\n")

        if score > best_score:
            best_model = model
            best_score = score

    print("\nTHE BEST MODEL IS ", str(best_model)," with a score of " + str(best_score)+". IT WILL BE USED TO GENERATE THE FINAL RESULT\n")
    gbc = GradientBoostingClassifier(learning_rate=best_model["learning_rate"], n_estimators=int(best_model["n_estimators"]), max_depth=int(best_model["max_depth"]), 
                                     min_samples_split=int(best_model["min_samples_split"]), min_samples_leaf=int(best_model["min_samples_leaf"]), random_state=42)
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    
    return y_pred


if __name__ == "__main__":

    preprocessing = True

    if preprocessing:
        t0 = time.time()
        X_train, X_test, y_train = data_preprocessing()
        t1 = time.time() 
        print("\nELAPSED MINUTES:", (t1-t0)/60, "\n")
    else:
        print("\nLOADING TRAINING DATA\n")
        y_train = pd.read_csv("./AdvancedML/task2/data/y_train.csv")
        y_train = y_train.drop('id', axis=1).to_numpy().flatten()
        X_train = np.loadtxt("./AdvancedML//task2/data/features_train_g3.csv", delimiter=",")
        print("\nLOADING TEST DATA\n")
        X_test = np.loadtxt("./AdvancedML//task2/data/features_train_g3.csv", delimiter=",") 


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
    
