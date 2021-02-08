# Automatic Speech Intelligibility Prediction From Single Channel Speech Signals

This repository contains the code for:
- dataset generation: add noise to clean signals [LibriSpeech](https://www.openslr.org/12) dataset;
- dataset generation: compute reliability/intelligibility index by comparing automatic transcription and original text;
- feature extraction: compute a set of audio features (MFCC, ZCR, SC, ...) 
- model training: train supervised classifiers for intelligibility prediction 


## References
"Automatic Reliability Estimation for Speech Audio Surveillance Recordings"
Clara Borrelli, Paolo Bestagini, Fabio Antonacci, Augusto Sarti, Stefano Tubaro
2019 IEEE International Workshop on Information Forensics and Security (WIFS)