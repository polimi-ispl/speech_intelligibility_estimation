import speech_recognition as sr
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


def transcribe(song_path, transcriber):
    # if file does not exist
    dest_path = song_path.replace('dataset', transcriber)
    dest_path = dest_path.replace(dest_path.split('.')[-1], 'txt')

    if os.path.exists(dest_path):
        return
    r = sr.Recognizer()
    print(song_path)
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
        text_file.write(transcription)  # TODO: controlla
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

    dataset_name = 'train-clean-360'
    in_csv_path = '/nas/home/cborrelli/speech_forensics/csv/' + dataset_name + '.csv'
    trans = 'sphinx'
    dataset_transcription(in_csv_path, trans)
