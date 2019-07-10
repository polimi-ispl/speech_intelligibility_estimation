import warnings
warnings.simplefilter('ignore', category=FutureWarning)
import numpy as np
import sklearn.model_selection
import sklearn.ensemble
import sklearn.svm
import sklearn.utils
import os
from tqdm import tqdm
import pandas as pd
pd.options.mode.chained_assignment = None


def df_to_feat_idx(feat_df, key, fun):
    X = feat_df.apply(lambda x: np.asarray(x[key]).squeeze().T[np.where(x['rms_idx'] == 1)].T, axis=1)
    X = X.apply(fun)
    X = X.apply(lambda x: np.reshape(x, (1, -1)))
    X = np.concatenate(X).squeeze()
    if len(X.shape) == 1:
        X.shape += (1,)
    return X


def train_one_configuration(norm, filter_outliers, rms_th, alg, win_min_max):

    res_file_name = 'res_norm-{}_outliers-{}_rms-{}_alg-{}_win-{}.pkl'.format(norm,
                                                                              filter_outliers,
                                                                              str(rms_th).replace('.', ''),
                                                                              alg,
                                                                              win_min_max)
    if os.path.exists(os.path.join('/nas/home/cborrelli/speech_forensics/results_add', res_file_name)):
        return


    # Load the features
    test_data_path = '/nas/home/cborrelli/speech_forensics/notebook/pickle/features_test-clean.pkl'
    train_data_path = '/nas/home/cborrelli/speech_forensics/notebook/pickle/features_train-clean-100.pkl'
    dev_data_path = '/nas/home/cborrelli/speech_forensics/notebook/pickle/features_dev-clean.pkl'

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
        feat_df = feat_df.loc[np.where(np.logical_and(feat_df['n_win'] >= n_win_min, feat_df['n_win'] <= n_win_max))[0]].reset_index()

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
    y_cl_multi = np.array(feat_df['y_label'], dtype=np.float) - 1  # labels for classification
    y_cl = np.array(feat_df['y_value'], dtype=np.float) >= 0.5  # labels for classification
    y_rg = np.array(feat_df['y_value'], dtype=np.float)  #  values for regression

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
        regr = sklearn.svm.SVR(gamma='auto')
    else:
        clf = sklearn.ensemble.RandomForestClassifier(class_weight='balanced', random_state=0)
        regr = sklearn.ensemble.RandomForestRegressor(random_state=0)

    # Train
    y_pred_cl = sklearn.model_selection.cross_val_predict(clf, X_norm, y_cl, cv=5, n_jobs=-1)
    y_pred_mcl = sklearn.model_selection.cross_val_predict(clf, X_norm, y_cl_multi, cv=5, n_jobs=-1)
    y_pred_r = sklearn.model_selection.cross_val_predict(regr, X_norm, y_rg, cv=5, n_jobs=-1)

    # Store results
    sel_columns = ['path', 'noise', 'snr', 'y_value', 'y_label', 'n_win', 'rms_idx']
    res_df = feat_df[sel_columns]
    res_df.loc[:, 'y_pred_cl'] = y_pred_cl
    res_df.loc[:, 'y_pred_mcl'] = y_pred_mcl
    res_df.loc[:, 'y_pred_r'] = y_pred_r

    # Save results
    res_df.to_pickle(os.path.join('/nas/home/cborrelli/speech_forensics/results_add', res_file_name))


if __name__ == '__main__':
    # Params
    param_dict = {'norm': ['nonorm'],
                  'filter_outliers': [True, False],
                  'rms_th': [0, 0.25, 0.50, 0.75],
                  'alg': ['svm', 'rf'],
                  'win_min_max': [True, False]
                  }

    # Generate experiments list
    param_list = sklearn.model_selection.ParameterGrid(param_dict)

    # Loop over experiments
    for params in tqdm(param_list):
        train_one_configuration(**params)
