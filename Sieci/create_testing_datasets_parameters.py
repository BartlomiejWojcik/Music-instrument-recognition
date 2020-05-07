import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import sys
import csv
import librosa.display
import IPython.display as ipd
import warnings
from pathlib import Path

path = Path(__file__)
warnings.filterwarnings('ignore')

genres = 'cel cla flu gac gel org pia sax tru vio voi'.split()

# defines some params out of .wav file
def specific_data(songname, g = ' '):
    y, sr = librosa.load(songname, mono=True, duration=30)
    # some parameters out of .wav file
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    to_append += f' {g}'
    return to_append.split()

# saves params into .csv files
def createCsvData():
    # create header
    header = 'chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    # write header
    with open('C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/datatest.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
    # write data for each audio
    for filename in os.listdir(f'C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/Music-instrument-recognition/Sieci/IRMAS-TestingData-Part1/Part1'):
        if (".txt"  in filename):
            f = open(f'C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/Music-instrument-recognition/Sieci/IRMAS-TestingData-Part1/Part1/{filename}', "r")
            genre_r = f.read()
            j = 0
            for i in range(int(len(genre_r)/5)):
                genre = genre_r[j] + genre_r[j+1] + genre_r[j+2]
                j = j + 5
                filename = filename[:-4] + ".wav"
                songname = f'C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/Music-instrument-recognition/Sieci/IRMAS-TestingData-Part1/Part1/{filename}'
                to_append = specific_data(songname, genre)
                file = open('C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/Music-instrument-recognition/Sieci/datatest.csv', 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append)

createCsvData()