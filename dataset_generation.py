import os
import pandas as pd
import speech_recognition as sr
import numpy as np
import scipy as sp
import scipy.io
import soundfile as sf
import librosa

from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


# 1) create csv for original dataset from Libri Speech

def create_csv_orig_dataset(orig_root, csv_dest_path):
    df_col = ['reader', 'book', 'librispeech_folder', 'path', 'transcription_path', 'SNR', 'noise_type']
    dir_df = pd.DataFrame(columns=df_col)
    for s in os.listdir(orig_root):
        for b in os.listdir(os.path.join(orig_root,s)):
            files = [ff for ff in os.listdir(os.path.join(orig_root, s ,b)) if ff.endswith('.flac')]
            trans = [tt for tt in os.listdir(os.path.join(orig_root, s, b)) if tt.endswith('.txt')][0]
            for f in files:
                row = [s, b, orig_root, os.path.join(orig_root, s, b, f), os.path.join(orig_root, s, b,trans), float('inf'), None]
                file_info = pd.Series(row, index=df_col)
                dir_df = dir_df.append(file_info, ignore_index = True)
    dir_df.to_csv(csv_dest_path, index=False)
    return



# 2) Transcribe the dataset using a specific transciber (only Sphinx is implemented)

def transcribe(song_path, transcriber):
    # if file does not exist
    dest_path = song_path.replace('dataset', transcriber)
    dest_path = dest_path.replace(dest_path.split('.')[-1], 'txt')

    r = sr.Recognizer()
    with sr.AudioFile(song_path) as source:
        audio = r.record(source)  # read the entire audio file

    # here depending on transcriber argument we will switch on different methods
    if transcriber == 'sphinx':
        transcription = r.recognize_sphinx(audio)
    else:
        transcription = ''

    try:
        os.makedirs(os.path.dirname(dest_path))
    except:
        pass

    with open(dest_path, "w") as text_file:
        text_file.write(transcription)
    return


def dataset_transcription(dataset_csv_path, transcriber):
    orig_df = pd.read_csv(dataset_csv_path)
    audio_path_list = orig_df['path'].to_list()
    f_part = partial(transcribe, transcriber=transcriber)
    pool = Pool(cpu_count())
    for _ in tqdm(pool.imap_unordered(f_part, audio_path_list), total=len(audio_path_list)):
        pass
    pool.close()

# 3) Select from the original dataset only audio with reliable transcription


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def select_reliable_trans(orig_csv_path, dest_csv_path, transcriber):
    orig_db = pd.read_csv(orig_csv_path)
    selected_db = pd.DataFrame(columns=orig_db.columns)

    for index, row in orig_db.iterrows():
        orig_name = row['path'].split('/')[-1].replace('.flac', '')
        real_text_path = row['transcription_path']
        with open(real_text_path) as f:
            content = f.readlines()
        real_text = []
        for rr in content:
            splitted = rr.split(' ')
            if splitted[0] == orig_name:
                real_text = ' '.join(splitted[1:])
                break
        real_text = real_text.lower().rstrip()

        trans_path = row['path'].replace('dataset', transcriber)
        trans_path = trans_path.replace(trans_path.split('.')[-1], 'txt')
        with open(trans_path, "r") as text_file:
            transcription = text_file.read()

        sim = get_jaccard_sim(transcription, real_text)

        if sim > 0.975:
            selected_db = selected_db.append(row, ignore_index=True)

    selected_db.to_csv(dest_csv_path, index=False)
    return


# 4) Add noise to the selected dataset

def add_noise(in_path, dataset_name, noise_type, snr):
    # generate output file path
    dest_path = in_path.replace(dataset_name, '{:s}_{:s}_{:d}'.format(dataset_name, noise_type, snr))

    # rimuovere
    dest_path = dest_path.replace('dev-clean', 'dev-clean_reduced')
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
    else:
        raise Exception('Noise sampling rate is too low.')

        # trim the noise
    if (noise.shape[0] > clean.shape[0]):
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
    noisy = 0.5*(clean + noise_scaled)

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
    pool.join()
    return

# 5) create csv for modified dataset given a transcriber?


def create_csv_modified_dataset(mod_root, csv_dest_path, transcriber, noise_type, snr):
    df_col = ['reader', 'book', 'librispeech_folder', 'path', 'transcription_path', 'SNR', 'noise_type']
    dir_df = pd.DataFrame(columns=df_col)

    for s in os.listdir(mod_root):
        for b in os.listdir(os.path.join(mod_root, s)):
            files = [ff for ff in os.listdir(os.path.join(mod_root, s, b))]
            for f in files:
                trans_path = os.path.join(mod_root, s, b, f).replace('dataset', transcriber)
                trans_path = trans_path.replace(trans_path.split('.')[-1], 'txt')
                row = [s, b, mod_root, os.path.join(mod_root, s, b, f), trans_path, snr, noise_type]
                file_info = pd.Series(row, index=df_col)
                dir_df = dir_df.append(file_info, ignore_index=True)
    dir_df.to_csv(csv_dest_path, index=False)
    return

# 6) call again transciber on the modified dataset