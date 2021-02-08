import speech_recognition as sr
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import argparse


def transcribe(song_path, transcriber):
    # if file does not exist
    dest_path = song_path.replace('dataset', transcriber)
    dest_path = dest_path.replace(dest_path.split('.')[-1], 'txt')

    if os.path.exists(dest_path):
        return
    r = sr.Recognizer()
    with sr.AudioFile(song_path) as source:
        audio = r.record(source)  # read the entire audio file

    # here depending on transcriber argument we will switch on different methods
    if transcriber == 'sphinx':
        try:
            transcription = r.recognize_sphinx(audio)
        except:
            transcription = ''
    elif transcriber == 'google':
        try:
            transcription = r.recognize_google(audio).lower()
        except:
            transcription = ''
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
    return


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)

    args = parser.parse_args()
    dataset_name = args.dataset_name

    # parameters
    # dataset_list = ['dev-clean', 'test-clean', 'train-clean-100']
    noise_type_list = ['train', 'hall', 'restaurant', 'car', 'industrial', 'crowd', 'traffic', 'speech1', 'speech2',
                       'speech3']
    snr_list = [0, 2, 5, 7, 10, 12, 15]
    trans = 'sphinx'
    #trans = 'google'

    # main loop
    for n in noise_type_list:
        for s in snr_list:

            in_csv_path = '/nas/home/cborrelli/speech_forensics/csv/sphinx/{:s}_{:s}_{:d}.csv'.format(dataset_name, n, s)
            dataset_transcription(in_csv_path, trans)
