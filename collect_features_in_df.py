import numpy as np
import pandas as pd
import tqdm
import os
import argparse


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def collect_feat_in_df(dataset, transcriber):
    columns = ['path', 'noise', 'snr', 'mfcc', 'sfl', 'sc', 'sroff', 'zcr', 'rms', 'y_value', 'y_label']
    snr = [0, 2, 5, 7, 10, 12, 15]
    noise_type = ['train', 'hall', 'restaurant', 'car', 'industrial', 'crowd', 'traffic', 'speech1', 'speech2',
                  'speech3']
    bins = [0.0, 0.2, 0.4, 0.6, 0.8]
    feat_df = pd.DataFrame(columns=columns)

    if os.path.exists('/nas/home/cborrelli/speech_forensics/notebook/pickle/{:s}/features_{:s}.pkl'.format(transcriber, dataset)):
        return
    for n in noise_type:
        for s in tqdm.tqdm(snr, total=len(snr)):
            orig_csv_path = '/nas/home/cborrelli/speech_forensics/csv/{:s}/{:s}.csv'.format(transcriber, dataset)
            noise_csv_path = '/nas/home/cborrelli/speech_forensics/csv/{:s}/{:s}_{:s}_{:d}.csv'.format(transcriber, dataset, n, s)

            orig_db = pd.read_csv(orig_csv_path)
            noise_db = pd.read_csv(noise_csv_path)

            for index, row in noise_db.iterrows():
                noise_path = row['path']
                feature_path = noise_path.replace(noise_path.split('.')[-1], 'npy')
                feature_path = feature_path.replace('dataset', 'features')
                x = np.load(feature_path, allow_pickle=True)
                mfcc = x.item().get('mfcc')

                sfl = x.item().get('sfl')[0]
                sc = x.item().get('sc')[0]
                sroff = x.item().get('sroff')[0]
                zcr = x.item().get('zcr')[0]
                rms = x.item().get('rms')[0]

                orig_path = row['path'].replace(row['path'].split('/')[-4], dataset).replace('wav', 'flac')
                orig_row = orig_db[orig_db['path'] == orig_path]
                real_text_path = orig_row['transcription_path'].values[0]
                orig_name = orig_path.split('/')[-1].replace('.flac', '')

                with open(real_text_path) as f:
                    content = f.readlines()
                real_text = []
                for rr in content:
                    splitted = rr.split(' ')
                    if splitted[0] == orig_name:
                        real_text = ' '.join(splitted[1:])
                        break
                if real_text:
                    real_text = real_text.lower().rstrip()
                else:
                    real_text = ''

                trans_path = row['transcription_path']
                with open(trans_path, "r") as text_file:
                    transcription = text_file.read()

                sim = get_jaccard_sim(transcription, real_text)

                newrow = pd.DataFrame(
                    [[noise_path, n, s, [mfcc], sfl, sc, sroff, zcr, rms, sim, np.digitize(sim, bins)]],
                    columns=columns)
                feat_df = feat_df.append(newrow, ignore_index=True)
    feat_df.to_pickle('/nas/home/cborrelli/speech_forensics/notebook/pickle/{:s}/features_{:s}.pkl'.format(transcriber, dataset))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--transcriber', type=str, required=True)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    transcriber = args.transcriber

    collect_feat_in_df(dataset_name, transcriber)
