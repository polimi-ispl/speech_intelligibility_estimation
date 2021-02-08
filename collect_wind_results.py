import pandas as pd
import sklearn
import sklearn.metrics
import numpy as np
import os

if __name__ == '__main__':

    result_folder = '/nas/home/cborrelli/speech_forensics/results_windowing/'

    win_feature_length = [10, 12, 14, 16, 18, 20]

    columns = ['window_length', 'cl_balanced_accuracy', 'cl_f1_score', 'mcl_balanced_accuracy', 'mcl_f1',
               'r_R2', 'r_mae', 'maj_cl_balanced_accuracy', 'maj_cl_f1_score', 'maj_mcl_balanced_accuracy',
               'maj_mcl_f1_score']
    results = pd.DataFrame(columns=columns)

    for w in win_feature_length:
        res_file_name = 'results_windowlenght-' + str(w)
        res_df = pd.read_pickle(os.path.join(result_folder, res_file_name))

        y_cl_true = res_df['y_binlabel']
        cl_balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_cl_true, res_df['y_pred_cl'])
        cl_f1_score = sklearn.metrics.f1_score(y_cl_true, res_df['y_pred_cl'])

        y_mcl_true = np.array(res_df['y_label'], dtype=np.float)  # labels for classification
        mcl_balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_mcl_true, res_df['y_pred_mcl'])
        mcl_f1 = sklearn.metrics.f1_score(y_mcl_true, res_df['y_pred_mcl'], average='micro')

        y_r_true = np.array(res_df['y_value'], dtype=np.float)  #  values for regression
        r_R2 = sklearn.metrics.r2_score(y_r_true, res_df['y_pred_r'])
        r_mae = sklearn.metrics.mean_absolute_error(y_r_true, res_df['y_pred_r'])

        # majority voting
        paths = res_df['path']
        maj_y_cl_true = []
        maj_y_cl_pred = []

        maj_y_mcl_true = []
        maj_y_mcl_pred = []
        for a in paths:
            subset = res_df[res_df['path'] == a]
            maj_y_cl_true.append(subset['y_binlabel'].unique()[0])

            count = np.bincount(subset['y_pred_cl'])
            maj_y_cl_pred.append(bool(np.argmax(count)))

            maj_y_mcl_true.append(subset['y_label'].unique()[0])

            count = np.bincount(subset['y_pred_mcl'])
            maj_y_mcl_pred.append(float(np.argmax(count)))

        maj_cl_balanced_accuracy = sklearn.metrics.balanced_accuracy_score(maj_y_cl_true, maj_y_cl_pred)
        maj_cl_f1_score = sklearn.metrics.f1_score(maj_y_cl_true, maj_y_cl_pred)

        maj_mcl_balanced_accuracy = sklearn.metrics.balanced_accuracy_score(maj_y_mcl_true, maj_y_mcl_pred)
        maj_mcl_f1_score = sklearn.metrics.f1_score(maj_y_mcl_true, maj_y_mcl_pred, average='micro')

        newrow = pd.DataFrame([[w, cl_balanced_accuracy, cl_f1_score, mcl_balanced_accuracy, mcl_f1,
                                r_R2, r_mae, maj_cl_balanced_accuracy, maj_cl_f1_score, maj_mcl_balanced_accuracy,
                                maj_mcl_f1_score]],
                              columns=columns)
        results = results.append(newrow, ignore_index=True)

    results.to_pickle(os.path.join('/nas/home/cborrelli/speech_forensics/results_windowing', 'metrics.pkl'))
