import librosa
import numpy as np
import scipy as sp
import scipy.io.wavfile
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse
import soundfile as sf


def compute(in_path):
    dest_path = in_path.replace('dataset', 'features')
    extension = in_path.split('.')[-1]
    dest_path = dest_path.replace(in_path.split('.')[-1], 'npy')
    if os.path.exists(dest_path):
        return

    # read wav file
    if extension == 'wav':
        rate, sig = sp.io.wavfile.read(filename=in_path)
        sig = np.float32(sig) / np.float32(2 ** (16 - 1))

    else:
        sig, rate = sf.read(in_path)

    # features parameters
    feat_frame_length_sec = 0.05
    feat_hop_size_sec = 0.025

    frame_length = int(rate * feat_frame_length_sec)
    feat_hop_size = int(rate * feat_hop_size_sec)
    N_MFCC = 22

    # compute features
    mfcc = librosa.feature.mfcc(sig, sr=rate,n_mfcc=N_MFCC,n_fft=frame_length, hop_length=feat_hop_size)
    sc = librosa.feature.spectral_centroid(sig, sr=rate, n_fft=frame_length, hop_length=feat_hop_size)
    sfl = librosa.feature.spectral_flatness(sig, n_fft=frame_length, hop_length=feat_hop_size)
    sroff = librosa.feature.spectral_rolloff(sig, n_fft=frame_length, hop_length=feat_hop_size, sr=rate)
    zcr = librosa.feature.zero_crossing_rate(sig, frame_length=frame_length, hop_length=feat_hop_size)
    rms = librosa.feature.rms(sig,frame_length=frame_length, hop_length=feat_hop_size)

    # store features into dictionary
    features = {}
    features['mfcc'] = mfcc
    features['sc'] = sc
    features['sfl'] = sfl
    features['sroff'] = sroff
    features['zcr'] = zcr
    features['rms'] = rms

    # create destination directory
    try:
        os.makedirs(os.path.dirname(dest_path))
    except:
        pass

    # save features to disk
    np.save(dest_path, features)

    return


def compute_features(dataset_csv_path):
    orig_df = pd.read_csv(dataset_csv_path)
    audio_path_list = orig_df['path'].to_list()
    pool = Pool(cpu_count() // 2)
    for _ in tqdm(pool.imap_unordered(compute, audio_path_list), total=len(audio_path_list)):
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

    noise_type_list = ['train', 'hall', 'restaurant', 'car', 'industrial', 'crowd', 'traffic', 'speech1', 'speech2',
                       'speech3']
    snr_list = [0, 2, 5, 7, 10, 12, 15]

    # extract from clean dataset
    orig_csv_path = '/nas/home/cborrelli/speech_forensics/csv/sphinx/' + dataset_name + '.csv'
    compute_features(orig_csv_path)

    # main loop
    for n in noise_type_list:
        for s in snr_list:
            in_csv_path = '/nas/home/cborrelli/speech_forensics/csv/sphinx/{:s}_{:s}_{:d}.csv'.format(dataset_name, n, s)
            compute_features(in_csv_path)
