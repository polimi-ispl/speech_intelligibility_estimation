import numpy as np
import sklearn.model_selection
import sklearn.ensemble
import sklearn.svm
import sklearn.utils
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
from skimage.util.shape import view_as_windows
import pickle


def df_to_feat_idx_wind(feat_df, key, fun, window_length):
    X = np.asarray(feat_df[key].values)
    XX = []
    wind_number = []
    if key == 'mfcc':
        for i, x in enumerate(X):
            x = np.asarray(x).T.squeeze()
            xw = view_as_windows(x, window_shape=(window_length, x.shape[1]),
                                 step=(int(window_length * 0.5), x.shape[1])).squeeze()
            fun_xw = fun(xw).squeeze()
            XX.append(fun_xw)
            wind_number.append(fun_xw.shape[0])

        XX = np.asarray(np.concatenate(XX, axis=0))

    else:
        for i, x in enumerate(X):
            x = np.asarray(x).squeeze()
            xw = view_as_windows(x, window_shape=(window_length,), step=int(window_length * 0.5))
            fun_xw = fun(xw)
            fun_xw = fun_xw.reshape(-1, 1)
            XX.append(fun_xw)
            wind_number.append(fun_xw.shape[0])

        XX = np.asarray(np.concatenate(XX, axis=0))

    return XX, np.array(wind_number)


def test_best_configuration_windowing(window_length):
    norm = 'zscore'
    win_min_max = True
    rms_th = 0
    n_win_min = 20
    n_win_max = 250
    filter_outliers = True


    pkl_filename = '/nas/home/cborrelli/speech_forensics/models/win_cl.pkl'
    with open(pkl_filename, 'rb') as file:
        cl = pickle.load(file)

    pkl_filename = '/nas/home/cborrelli/speech_forensics/models/win_mcl.pkl'
    with open(pkl_filename, 'rb') as file:
        mcl = pickle.load(file)

    pkl_filename = '/nas/home/cborrelli/speech_forensics/models/win_r.pkl'
    with open(pkl_filename, 'rb') as file:
        r = pickle.load(file)

    test_data_path = '/nas/home/cborrelli/speech_forensics/notebook/pickle/sphinx/features_test-clean.pkl'
    feat_df = pd.read_pickle(test_data_path)
    feat_df = sklearn.utils.shuffle(feat_df, random_state=0).reset_index(drop=True)

    # feat_df['rms_idx'] = feat_df['rms'].apply(lambda x: x >= rms_th * (x.max() - x.min()) + x.min())
    feat_df['n_win'] = feat_df['rms'].apply(lambda x: len(x))
    if win_min_max:
        feat_df = feat_df.loc[
            np.where(np.logical_and(feat_df['n_win'] >= n_win_min, feat_df['n_win'] <= n_win_max))[0]].reset_index()

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

    key_list = ['mfcc', 'sfl', 'sc', 'sroff', 'zcr', 'rms']

    X_mean_list = []
    for key in key_list:
        X_m, _ = df_to_feat_idx_wind(feat_df, key, lambda x: np.mean(x, axis=1), window_length=window_length)
        X_mean_list += [X_m]

    X_mean = np.concatenate(X_mean_list, axis=1)

    X_std_list = []
    for key in key_list:
        X_s, _ = df_to_feat_idx_wind(feat_df, key, lambda x: np.std(x, axis=1), window_length=window_length)
        X_std_list += [X_s]
    X_std = np.concatenate(X_std_list, axis=1)

    X_max_list = []
    for key in key_list:
        X_m, _ = df_to_feat_idx_wind(feat_df, key, lambda x: np.max(x, axis=1), window_length=window_length)
        X_max_list += [X_m]
    X_max = np.concatenate(X_max_list, axis=1)

    X_min_list = []
    for key in key_list:
        X_m, wind_number = df_to_feat_idx_wind(feat_df, key, lambda x: np.min(x, axis=1),
                                               window_length=window_length)
        X_min_list += [X_m]
    X_min = np.concatenate(X_min_list, axis=1)

    X = np.concatenate([X_mean, X_std, X_max, X_min], axis=1)

    y_mcl = np.array(feat_df['y_label'], dtype=np.float) - 1  # labels for classification
    y_cl = np.array(feat_df['y_value'], dtype=np.float) >= 0.5  # labels for classification
    y_rg = np.array(feat_df['y_value'], dtype=np.float)

    y_mcl = np.repeat(y_mcl, wind_number)
    y_cl = np.repeat(y_cl, wind_number)
    y_rg = np.repeat(y_rg, wind_number)

    snrs = feat_df['snr']
    snrs = [item for item, count in zip(snrs, wind_number) for i in range(count)]

    noises = feat_df['noise']
    noises = [item for item, count in zip(noises, wind_number) for i in range(count)]

    paths = feat_df['path']
    paths = [item for item, count in zip(paths, wind_number) for i in range(count)]

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

    y_pred_cl = cl.predict(X_norm)
    y_pred_mcl = mcl.predict(X_norm)
    y_pred_r = r.predict(X_norm)

    columns = ['path','noise', 'snr', 'y_binlabel', 'y_value', 'y_label', 'y_pred_cl', 'y_pred_mcl', 'y_pred_r']
    res_df = pd.DataFrame(columns=columns)
    res_df.loc[:, 'path'] = paths
    res_df.loc[:, 'noise'] = noises
    res_df.loc[:, 'snr'] = snrs
    res_df.loc[:, 'y_binlabel'] = y_cl
    res_df.loc[:, 'y_value'] = y_rg
    res_df.loc[:, 'y_label'] = y_mcl
    res_df.loc[:, 'y_pred_cl'] = y_pred_cl
    res_df.loc[:, 'y_pred_mcl'] = y_pred_mcl
    res_df.loc[:, 'y_pred_r'] = y_pred_r

    res_file_name = 'results_windowlenght-' + str(window_length)
    res_df.to_pickle(os.path.join('/nas/home/cborrelli/speech_forensics/results_windowing', res_file_name))
    return


if __name__ == '__main__':
    windows = [10, 12, 14, 16, 18, 20]
    pool = Pool(cpu_count() // 2)
    for _ in tqdm(pool.imap_unordered(test_best_configuration_windowing
, windows), total=len(windows)):
        pass
    pool.close()

