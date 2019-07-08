import warnings

warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)
import numpy as np
import sklearn.model_selection
import sklearn.ensemble
import sklearn.svm
import sklearn.utils
import os
from tqdm import tqdm
import pandas as pd
from skimage.util.shape import view_as_windows

pd.options.mode.chained_assignment = None



def df_to_feat_idx_wind(feat_df, key, fun, window_length, return_label=False):
    X = np.asarray(feat_df[key].values)
    Y_v = np.asarray(feat_df['y_value'])
    Y_l = np.asarray(feat_df['y_label'])
    XX = []
    YY_v = []
    YY_l = []
    if key == 'mfcc':
        for i, x in enumerate(X):
            x = np.asarray(x).T.squeeze()
            xw = view_as_windows(x, window_shape=(window_length, x.shape[1]),
                                 step=(int(window_length * 0.5), x.shape[1])).squeeze()
            fun_xw = fun(xw).squeeze()
            XX.append(fun_xw)
            if return_label:
                YY_v.append(np.repeat(Y_v[i], fun_xw.shape[0]))
                YY_l.append(np.repeat(Y_l[i], fun_xw.shape[0]))

        XX = np.asarray(np.concatenate(XX, axis=0))
        if return_label:
            YY_v = np.asarray(np.concatenate(YY_v, axis=0))
            YY_l = np.asarray(np.concatenate(YY_l, axis=0))
    else:
        for i, x in enumerate(X):
            x = np.asarray(x).squeeze()
            xw = view_as_windows(x, window_shape=(window_length,), step=int(window_length * 0.5))
            fun_xw = fun(xw)
            fun_xw = fun_xw.reshape(-1, 1)
            XX.append(fun_xw)
            if return_label:
                YY_v.append(np.repeat(Y_v[i], fun_xw.shape[0]))
                YY_l.append(np.repeat(Y_l[i], fun_xw.shape[0]))

        XX = np.asarray(np.concatenate(XX, axis=0))
        if return_label:
            YY_v = np.asarray(np.concatenate(YY_v, axis=0))
            YY_l = np.asarray(np.concatenate(YY_l, axis=0))

    return XX, YY_v, YY_l


def train_one_configuration(norm, filter_outliers, rms_th, alg, win_min_max, win_feature_length):
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
        X_m, _, _ = df_to_feat_idx_wind(feat_df, key, lambda x: np.mean(x, axis=1), window_length=win_feature_length)
        X_mean_list += [X_m]

    X_mean = np.concatenate(X_mean_list, axis=1)

    X_std_list = []
    for key in key_list:
        X_s, _, _ = df_to_feat_idx_wind(feat_df, key, lambda x: np.std(x, axis=1), window_length=win_feature_length)
        X_std_list += [X_s]
    X_std = np.concatenate(X_std_list, axis=1)

    X_max_list = []
    for key in key_list:
        X_m, _, _ = df_to_feat_idx_wind(feat_df, key, lambda x: np.max(x, axis=1), window_length=win_feature_length)
        X_max_list += [X_m]
    X_max = np.concatenate(X_max_list, axis=1)

    X_min_list = []
    for key in key_list:
        X_m, y_rg, y_cl_multi = df_to_feat_idx_wind(feat_df, key, lambda x: np.min(x, axis=1),
                                                    window_length=win_feature_length, return_label=True)
        X_min_list += [X_m]
    X_min = np.concatenate(X_min_list, axis=1)

    X = np.concatenate([X_mean, X_std, X_max, X_min], axis=1)

    y_cl = y_rg >= 0.5


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
    res_file_name = 'res_norm-{}_outliers-{}_rms-{}_alg-{}_win-{}-winfeat-{}.pkl'.format(norm,
                                                                              filter_outliers,
                                                                              str(rms_th).replace('.', ''),
                                                                              alg,
                                                                              win_min_max,
                                                                              str(win_feature_length))
    res_df.to_pickle(os.path.join('/nas/home/cborrelli/speech_forensics/results', res_file_name))


if __name__ == '__main__':
    # Params
    param_dict = {'norm': ['zscore'],
                  'filter_outliers': [True, False],
                  'rms_th': [0],
                  'alg': ['svm', 'rf'],
                  'win_min_max': [True, False],
                  'win_feature_length': [10, 12, 14, 16, 18, 20]
                  }

    # Generate experiments list
    param_list = sklearn.model_selection.ParameterGrid(param_dict)

    # Loop over experiments
    for params in tqdm(param_list):
        train_one_configuration(**params)
