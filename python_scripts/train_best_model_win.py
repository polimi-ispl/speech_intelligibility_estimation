import warnings
warnings.simplefilter('ignore', category=FutureWarning)
import numpy as np
import sklearn.model_selection
import sklearn.ensemble
import sklearn.svm
import sklearn.utils
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None
import sklearn.model_selection
import sklearn.ensemble
import sklearn.svm
import sklearn.utils
import os
import pandas as pd
base_path = '/Users/Clara/Desktop/Dottorato/AudioForensics/projects/speech_forensics'


def df_to_feat_idx(feat_df, key, fun):
    X = feat_df.apply(lambda x: np.asarray(x[key]).squeeze().T[np.where(x['rms_idx'] == 1)].T, axis=1)
    X = X.apply(fun)
    X = X.apply(lambda x: np.reshape(x, (1, -1)))
    X = np.concatenate(X).squeeze()
    if len(X.shape) == 1:
        X.shape += (1,)
    return X


def train_one_configuration(norm, filter_outliers, rms_th, alg, win_min_max, num_classes):

    # Load the features
    test_data_path = base_path + '/notebook/pickle/sphinx/features_test-clean.pkl'
    train_data_path = base_path + '/notebook/pickle/sphinx/features_train-clean-100.pkl'
    dev_data_path = base_path + '/notebook/pickle/sphinx/features_dev-clean.pkl'

    feat_train_df = pd.read_pickle(train_data_path)
    feat_train_df['dataset'] = 'train'
    feat_test_df = pd.read_pickle(test_data_path)
    feat_test_df['dataset'] = 'test'
    feat_dev_df = pd.read_pickle(dev_data_path)
    feat_dev_df['dataset'] = 'dev'
    feat_df = pd.concat([feat_train_df, feat_test_df, feat_dev_df], ignore_index=True)

    # Shuffle the dataset
    feat_df = sklearn.utils.shuffle(feat_df, random_state=0).reset_index(drop=True)

    # Apply rms threshold
    feat_df['rms_idx'] = feat_df['rms'].apply(lambda x: x >= rms_th * (x.max() - x.min()) + x.min())
    feat_df['n_win'] = feat_df['rms'].apply(lambda x: len(x))

    # Apply threshold on window number
    if win_min_max:
        n_win_min = 50
        n_win_max = 250
        feat_df = feat_df.loc[
            np.where(np.logical_and(feat_df['n_win'] >= n_win_min, feat_df['n_win'] <= n_win_max))[0]].reset_index()

    # Filter out outliers
    if filter_outliers:
        idx_0 = (feat_df['y_value'] >= 0) & (feat_df['y_value'] < 0.35) & (feat_df['snr'] == 0)
        idx_2 = (feat_df['y_value'] >= 0) & (feat_df['y_value'] < 0.4) & (feat_df['snr'] == 2)
        idx_5 = (feat_df['y_value'] >= 0.1) & (feat_df['y_value'] < 0.5) & (feat_df['snr'] == 5)
        idx_7 = (feat_df['y_value'] >= 0.1) & (feat_df['y_value'] < 0.7) & (feat_df['snr'] == 7)
        idx_10 = (feat_df['y_value'] >= 0.25) & (feat_df['y_value'] < 0.7) & (feat_df['snr'] == 10)
        idx_12 = (feat_df['y_value'] >= 0.35) & (feat_df['y_value'] < 1.1) & (feat_df['snr'] == 12)
        idx_15 = (feat_df['y_value'] >= 0.4) & (feat_df['y_value'] < 1.1) & (feat_df['snr'] == 15)

        idx = idx_0 | idx_2 | idx_5 | idx_7 | idx_10 | idx_12 | idx_15

        feat_df = feat_df.loc[np.where(idx == 1)].reset_index()

    # Compute feature matrix
    key_list = ['mfcc', 'sfl', 'sc', 'sroff', 'zcr', 'rms']

    X_mean_list = []
    for key in key_list:
        X_mean_list += [df_to_feat_idx(feat_df, key, lambda x: np.mean(x, axis=-1))]
    X_mean = np.concatenate(X_mean_list, axis=1)

    X_std_list = []
    for key in key_list:
        X_std_list += [df_to_feat_idx(feat_df, key, lambda x: np.std(x, axis=-1))]
    X_std = np.concatenate(X_std_list, axis=1)

    X_max_list = []
    for key in key_list:
        X_max_list += [df_to_feat_idx(feat_df, key, lambda x: np.max(x, axis=-1))]
    X_max = np.concatenate(X_max_list, axis=1)

    X_min_list = []
    for key in key_list:
        X_min_list += [df_to_feat_idx(feat_df, key, lambda x: np.min(x, axis=-1))]
    X_min = np.concatenate(X_min_list, axis=1)

    X = np.concatenate([X_mean, X_std, X_max, X_min], axis=1)

    # Retrieve labels
    #y_cl_multi = np.array(feat_df['y_label'], dtype=np.float) - 1  # labels for classification
    y_cl = np.array(feat_df['y_value'], dtype=np.float) >= 0.5  # labels for classification
    y_rg = np.array(feat_df['y_value'], dtype=np.float)  #  values for regression

    bins = np.arange(0, 1, 1/num_classes)
    y_cl_multi = np.array(np.digitize(y_rg, bins) - 1, dtype=np.float)


    # Normalize features
    if norm == 'zscore':
        X_norm = (X - X.mean(axis=0)) / X.std(axis=0)  # z-score
    elif norm == 'minmax':
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # [0, 1]
    else:
        X_norm = X

    # Remove nan and inf
    X_norm[np.where(np.isnan(X_norm))] = 0
    X_norm[np.where(np.isinf(X_norm))] = 0

    # Select algorithm
    if alg == 'svm':
        clf = sklearn.svm.SVC(kernel='rbf', gamma='auto', random_state=0)
        clf_bin = sklearn.svm.SVC(kernel='rbf', gamma='auto', random_state=0)
        regr = sklearn.svm.SVR(gamma='auto')
    else:
        clf = sklearn.ensemble.RandomForestClassifier(class_weight='balanced', random_state=0)
        clf_bin = sklearn.svm.SVC(kernel='rbf', gamma='auto', random_state=0)
        regr = sklearn.ensemble.RandomForestRegressor(random_state=0)

    # Train
    #cl = clf_bin.fit(X_norm, y_cl)
    mcl = clf.fit(X_norm, y_cl_multi)
    #r = regr.fit(X_norm, y_rg)

    #out_filename = base_path + '/models/rai_r.pkl'
    #with open(out_filename, 'wb') as file:
    #    pickle.dump(r, file)

    #out_filename = base_path + '/models/rai_cl.pkl'
    #with open(out_filename, 'wb') as file:
    #    pickle.dump(cl, file)

    out_filename = base_path + '/models/rai_mcl.pkl'
    with open(out_filename, 'wb') as file:
        pickle.dump(mcl, file)

    return


if __name__ == '__main__':
    norm = 'zscore'
    filter_outliers = True
    win_min_max = True
    rms_th = 0
    alg = 'svm'
    num_classes = 3


    train_one_configuration(norm, filter_outliers, rms_th, alg, win_min_max, num_classes)