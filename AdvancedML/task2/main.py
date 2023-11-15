"""
The following script solves task 2 of the Advanced Machine Learning 2023 course at ETH Zurich.
TODO: more detailed explanation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score #F1 = f1_score(y_true, y_pred, average='micro')
from biosppy.signals.ecg import ecg #TODO: maybe try another package, see this post: https://www.samproell.io/posts/signal/ecg-library-comparison/
import matplotlib.pyplot as plt
import time

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
    R_index, output["R_signal"], output["R_time"] = find_extreme(template, template_ts, 0, "right", "max", a=-0.1, b=0.1 )

    # S-peak is the smallest signal to the right of the R-peak in a well behaved template
    S_index, output["S_signal"], output["S_time"] = find_extreme(template, template_ts, R_index, "right", "min", a=0, b=0.12)

    # Q-peak is the smallest signal to the left of the R-peak in a well behaved template
    Q_index, output["Q_signal"], output["Q_time"] = find_extreme(template, template_ts, R_index, "left", "min", a=-0.12, b=0)

    # T-peak is the largest signal to the right of the S-peak in a well behaved template
    T_index, output["T_signal"], output["T_time"] = find_extreme(template, template_ts, S_index, "right", "max", a=0.12)

    # P-peak is the largest signal to the left of the Q-peak in a well behaved template
    P_index, output["P_signal"], output["P_time"] = find_extreme(template, template_ts, Q_index, "left", "max", b=-0.12)

    # fill signal pairs
    for i, n in enumerate(["P", "Q", "R", "S", "T"]):
        for j, m in enumerate(["P", "Q", "R", "S", "T"]):
            if i>=j: 
                continue
            output[n + "-" + m + "_signal"] = output[n + "_signal"] - output[m + "_signal"]

    R_right_time, R_left_time, output["R_duration"] = get_wave_duration(template, template_ts, R_index)
    P_right_time, P_left_time, output["P_duration"] = get_wave_duration(template, template_ts, P_index)
    Q_right_time, Q_left_time, output["Q_duration"] = get_wave_duration(template, template_ts, Q_index)
    S_right_time, S_left_time, output["S_duration"] = get_wave_duration(template, template_ts, S_index)
    T_right_time, T_left_time, output["T_duration"] = get_wave_duration(template, template_ts, T_index)

    # fill pairwise wave durations. it is unclear whether for example to go from start/end of P wave to start/end of T wave.
    # The medical literature has no systematic nomenclature, however all technical medical quantities are used below with their 
    # name in the literature (see comments). Otherwise, I apply the convention that for example the "PR duration" is the difference
    # between the start of R and the start of P

    output["PR-interval"] = Q_left_time - P_right_time #technical medical term, called "PR" even though QP is concerned
    output["PR-segment"] = Q_left_time - P_left_time #technical medical term called "PR" even though QP is concerned
    output["PR-duration"] = R_left_time - P_left_time
    output["PS-duration"] = S_left_time - P_left_time
    output["total-duration"] = T_right_time - P_left_time #not a technical term, describes the total extent of the waveform
    output["VAT"] = output["R_time"] - Q_left_time  #technical medical term: ventricular activation time
    output["QRS-duration"] = S_right_time - Q_left_time #technical medical term
    output["QT-duration"] = T_left_time - Q_left_time
    output["VDT"] = S_right_time - output["R_time"] # not a technical term but defined by me analogous to ventricular activation time: ventricular deactivation time
    output["RT-duration"] = T_left_time - R_left_time
    output["ST-segment"] = T_left_time - S_right_time #technical medical term
    output["ST-interval"] = T_right_time - S_right_time #technical medical term

    if False:
        plt.figure()
        plt.plot(template_ts, template, '-r')
        plt.plot([output["P_time"], output["Q_time"], output["R_time"], output["S_time"], output["T_time"]], [output["P_signal"], output["Q_signal"], output["R_signal"], output["S_signal"], output["T_signal"]],'xb')
        plt.plot([P_right_time, P_left_time], [0,0],'ok')
        plt.plot([Q_right_time, Q_left_time], [0,0],'ok')
        plt.plot([R_right_time, R_left_time], [0,0],'ok')
        plt.plot([S_right_time, S_left_time], [0,0],'ok')
        plt.plot([T_right_time, T_left_time], [0,0],'ok')
        plt.show()

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
    #for 6 samples ecg returns an empty heartrate, which makes no sense
    #TODO: calculate it yourself using rpeaks
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg(signal=sample[~np.isnan(sample)], sampling_rate=300, show = False)

    template_features = []
    for i in range(len(templates)):
        template_features.append(get_PQRST(templates[i], templates_ts))

    template_features = pd.DataFrame(template_features)
    #[60/x for x in np.diff([ts[rpeak] for rpeak in rpeaks])]
    output = {}
    for col in template_features.columns:
        x = template_features[col].to_numpy()
        output[col+"_median"] = np.median(x)
        output[col+"_MAD"] = np.median(np.abs((x-output[col+"_median"])))

        output["heartrate_median"] = np.median(heart_rate)
        # ecg outputs no heartrate if it is very low, which leads to nans, derive your own proxy for the heart rate here
        if np.isnan(output["heartrate_median"]):
            heart_rate_proxy = [60/x for x in np.diff([ts[rpeak] for rpeak in rpeaks])]
            output["heartrate_median"] = np.median(heart_rate_proxy)
            output["heartrate_MAD"]= np.median(np.abs((heart_rate_proxy - output["heartrate_median"])))
        else:  
            output["heartrate_MAD"]= np.median(np.abs((heart_rate - output["heartrate_median"]))) 

    if False and sample_index<60 and sample_index>40:
        print("\nP: ", output["P_time_MAD"])
        print("\nQ: ", output["Q_time_MAD"])
        print("\nR: ", output["R_time_MAD"])
        print("\nS: ", output["S_time_MAD"])
        print("\nT: ", output["T_time_MAD"])
        plt.figure()
        plt.plot(templates_ts, np.median(templates, axis=0), '-r')
        plt.plot([output["P_time_median"], output["Q_time_median"], output["R_time_median"], output["S_time_median"], output["T_time_median"]], [output["P_signal_median"], output["Q_signal_median"], output["R_signal_median"], output["S_signal_median"], output["T_signal_median"]],'xk' )
        plt.plot(np.linspace(output["P_time_median"]-output["P_duration_median"]/2,output["P_time_median"]+output["P_duration_median"]/2,100), [0 for i in range(100)],'-k')
        plt.plot(np.linspace(output["Q_time_median"]-output["Q_duration_median"]/2,output["Q_time_median"]+output["Q_duration_median"]/2,100), [0 for i in range(100)],'-b')

        plt.plot(np.linspace(output["S_time_median"]-output["S_duration_median"]/2,output["S_time_median"]+output["S_duration_median"]/2,100), [0 for i in range(100)],'-k')
        plt.plot(np.linspace(output["T_time_median"]-output["T_duration_median"]/2,output["T_time_median"]+output["T_duration_median"]/2,100), [0 for i in range(100)],'-k')
        
        plt.show()

    return output



def data_preprocessing():
    """
    TODO: write docstring
    Parameters
    ----------

    Returns
    ----------
   
    """

    # Load training features and labels
    print("\nLOADING TRAINING DATA\n")
    X_train = pd.read_csv("./AdvancedML/task2/data/X_train.csv")
    #X_train = pd.read_csv("./AdvancedML/task2/data/X_train_small.csv")
    #X_train.iloc[:100,:].to_csv("./AdvancedML/task2/data/X_train_small.csv", header=True, index=False)
    y_train = pd.read_csv("./AdvancedML/task2/data/y_train.csv")
    # Load test features
    print("\nLOADING TEST DATA\n")
    X_test = pd.read_csv("./AdvancedML/task2/data/X_test.csv")

    X_train = X_train.drop('id', axis=1).to_numpy()
    y_train = y_train.drop('id', axis=1).to_numpy()
    X_test = X_test.drop('id', axis=1).to_numpy()

    print("\nEXTRACTING TRAINING FEATURES\n")
    features_train = list(np.apply_along_axis(get_features, 1, X_train))
    features_train = pd.DataFrame(features_train)
    features_train = features_train.drop(["P_time_median", "Q_time_median", "R_time_median", "S_time_median", "T_time_median", 
                              "P_time_MAD", "Q_time_MAD", "R_time_MAD", "S_time_MAD", "T_time_MAD"], axis=1)
    features_train = features_train.to_numpy()

    print("\nEXTRACTING TEST FEATURES\n") 
    features_test = list(np.apply_along_axis(get_features, 1, X_test))
    features_test = pd.DataFrame(features_test)
    features_test = features_test.drop(["P_time_median", "Q_time_median", "R_time_median", "S_time_median", "T_time_median", 
                              "P_time_MAD", "Q_time_MAD", "R_time_MAD", "S_time_MAD", "T_time_MAD"], axis=1)
    features_test = features_test.to_numpy()

    print("\nSCALING DATA\n")
    scaler = StandardScaler()
    scaler.fit(np.vstack([features_train, features_test]))
    features_train_scaled = scaler.transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    print("\nREDUCING DIMENSIONALITY\n")
    pca = PCA(n_components=0.9)
    pca.fit(np.vstack([features_train_scaled, features_test_scaled]))
    features_train_pca = pca.transform(features_train_scaled)
    features_test_pca = pca.transform(features_test_scaled)
    print("\nNumber of kept components: ", np.shape(features_train_pca)[1])

    # TODO: deal with class imbalance: either by resampling or using an SVM with class_weights = "balanced"

    np.savetxt("./AdvancedML//task2/data/features_train_09pca_indTemp_StdScaler.csv", features_train_pca, delimiter=",")
    np.savetxt("./AdvancedML//task2/data/feautres_test_09pca_indTemp_StdScaler.csv", features_test_pca, delimiter=",")

    return features_train_pca, features_test_pca, y_train


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

    t0 = time.time()
    X_train, X_test, y_train = data_preprocessing()
    t1 = time.time() 
    print("\nELAPSED MINUTES:", (t1-t0)/60, "\n") 

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
