import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import librosa.display
import IPython.display as ipd
import warnings
warnings.filterwarnings('ignore')

genres = 'cel cla flu gac gel org pia sax tru vio voi'.split()

# create specgrams and save it in appropriate directories
def create_specgrams():
    cmap = plt.get_cmap('inferno')
    for g in genres:
        for filename in os.listdir(f'C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/IRMAS-TrainingData/{g}'):
            name = f'C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/IRMAS-TrainingData/{g}/{filename}'
            y, sr = librosa.load(name, mono=True, duration=5)
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
            plt.show()
            plt.axis('off');
            plt.savefig(f'C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/IRMAS-TrainingDataPng/{g}/{filename[:-3].replace(".", "")}.png')
            plt.clf()

def create_specgrams_patki():
    for g in genres:
        for filename in os.listdir(f'C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/IRMAS-TrainingData/{g}'):
            audio_path = 'C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/IRMAS-TrainingData/cel/[cel][cla]0001__1.wav'
            x, sr = librosa.load(audio_path)
            hop_length = 512
            n_mels = 128
            n_fft = 2048
            S = librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_DB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length);
            plt.savefig(f'C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/IRMAS-TrainingDataPngPatki/{g}/{filename[:-3].replace(".", "")}.png')
            plt.clf()

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
    with open('C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
    # write data for each audio
    for g in genres:
        for filename in os.listdir(f'C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/IRMAS-TrainingData/{g}'):
            songname = f'C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/IRMAS-TrainingData/{g}/{filename}'
            to_append = specific_data(songname, g)
            file = open('C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append)

def createDataToNetwork():
    data = pd.read_csv('C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/data.csv')
    # last calumn (instruments)
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)
    # all columns despite last column
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
    #X = np.array(data.iloc[:, :-1], dtype=float)
    return X, y

def predictWavFile(path, model):
    to_append = specific_data(path)
    to_append = np.array(to_append, dtype = float)
    tab = np.zeros((1,25), dtype = float)
    #tab[0] = to_append
    scaler = StandardScaler()
    X = scaler.fit_transform(to_append.reshape(-1, 1))
    tab[0] = np.array(X[:1], dtype = float)
    return model.predict(tab[:1])
    #return model.predict(X[:1])