
import speech_recognition as sr
import numpy as np
import scipy as sp
import scipy.io
import soundfile as sf
import librosa
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import argparse


def create_csv_modified_dataset(mod_root, csv_dest_path, transcriber, noise_type, snr):
    df_col = ['reader', 'book', 'librispeech_folder', 'path', 'transcription_path', 'SNR', 'noise_type']
    dir_df = pd.DataFrame(columns=df_col)

    for s in os.listdir(mod_root):
        for b in os.listdir(os.path.join(mod_root, s)):
            files = [ff for ff in os.listdir(os.path.join(mod_root, s, b))]
            for f in files:
                lf = os.path.split(mod_root)[-1]
                trans_path = os.path.join(mod_root, s, b, f).replace('dataset', transcriber)
                trans_path = trans_path.replace(trans_path.split('.')[-1], 'txt')
                row = [s, b, lf, os.path.join(mod_root, s, b, f), trans_path, snr, noise_type]
                file_info = pd.Series(row, index=df_col)
                dir_df = dir_df.append(file_info, ignore_index=True)
    dir_df.to_csv(csv_dest_path, index=False)
    return


def add_noise(in_path, dataset_name, noise_type, snr):
    # generate output file path
    dest_path = in_path.replace(dataset_name, '{:s}_{:s}_{:d}'.format(dataset_name, noise_type, snr))
    dest_path = dest_path.replace(dest_path.split('.')[-1], 'wav')

    # check if file exists
    if os.path.exists(dest_path):
        return

    # read clean signal
    clean, samplerate = sf.read(in_path)
    clean = np.array(clean)

    # read and normalize noise
    noise_path = os.path.join('/nas/home/cborrelli/speech_forensics/dataset/noise', noise_type + '.wav')
    noise_samplerate, noise = scipy.io.wavfile.read(noise_path)
    noise = np.float32(noise) / np.float32(2 ** (16 - 1))

    # resample if necessary
    if samplerate < noise_samplerate:
        noise = librosa.resample(noise, noise_samplerate, samplerate)
    elif samplerate == noise_samplerate:
        pass
    else:
        raise Exception('Noise sampling rate is too low.')

        # trim the noise
    if noise.shape[0] > clean.shape[0]:
        noise = noise[:clean.shape[0]]
    else:
        raise Exception('Noise is too short.')

        # normalize to 0 dB nominal level
    norm_factor = np.sqrt(np.mean(np.abs(clean) ** 2) / np.mean(np.abs(noise) ** 2))
    noise = noise * norm_factor

    # apply scaling
    noise_gain = 1 / (10 ** (snr / 20))
    noise_scaled = noise * noise_gain

    # mix the two signals
    noisy = 0.5 * (clean + noise_scaled)
    noisy = np.clip(np.asarray(noisy*(2**15)), -2**15, 2**15).astype(np.int16)

    # save file to disk
    try:
        os.makedirs(os.path.dirname(dest_path))
    except:
        pass
    sp.io.wavfile.write(dest_path, samplerate, noisy)

    return


def add_noise_dataset(dataset_csv_path, noise_type, snr):
    orig_db = pd.read_csv(dataset_csv_path)
    in_path_list = orig_db['path'].to_list()
    dataset_name = orig_db['librispeech_folder'].unique()[0]

    f_part = partial(add_noise, dataset_name=dataset_name, noise_type=noise_type, snr=snr)

    pool = Pool(cpu_count())

    for _ in tqdm(pool.imap_unordered(f_part, in_path_list), total=len(in_path_list)):
        pass
    pool.close()
    return


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)

    args = parser.parse_args()
    dataset_name = args.dataset_name

    # parameters

    noise_type_list = ['train', 'hall', 'restaurant', 'car', 'industrial', 'crowd', 'traffic','speech1', 'speech2',
                       'speech3']
    snr_list = [0, 2, 5, 7, 10, 12, 15]

    # main loop
    in_dataset_csv = '/nas/home/cborrelli/speech_forensics/csv/sphinx/' + dataset_name + '_sel.csv'
    for n in noise_type_list:
        for s in snr_list:
            # generate noisy speech fragments
            add_noise_dataset(in_dataset_csv, n, s)

            # create the corresponding csv
            added_folder = '/nas/home/cborrelli/speech_forensics/dataset/{:s}_{:s}_{:d}'.format(dataset_name, n, s)
            out_dataset_csv = '/nas/home/cborrelli/speech_forensics/csv/sphinx/{:s}_{:s}_{:d}.csv'.format(dataset_name,
                                                                                                          n, s)
            create_csv_modified_dataset(added_folder, out_dataset_csv, 'sphinx', n, s)
