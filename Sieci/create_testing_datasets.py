import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import librosa.display
import IPython.display as ipd
import warnings
from pathlib import Path
from shutil import copyfile
import imageio

path = Path(__file__)

warnings.filterwarnings('ignore')

instruments_set = ('cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi')
genres = 'cel cla flu gac gel org pia sax tru vio voi'.split()
testing_X = []
testing_y = []
i = 0

# TODO: Change audio duration to 3s 

def create_datasets():
    i = 0
    for filename in os.listdir(str(path.parent.parent) + f'\\IRMAS-TestingData-Part1\\Part1\\'):
        print(filename)
        audio_path = str(path.parent.parent) + f'\\IRMAS-TestingData-Part1\\Part1\\{filename}'
        i = i + 1
        if(filename[-3:] == 'wav'): 
            testing_X.append(get_spectrogram_data(audio_path))
        else:
            testing_y.append(get_categories(audio_path))

def get_categories(audio_path):
    with open(audio_path, 'r') as file:
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        data = file.readlines()
        for instrument in data:
            result[instruments_set.index(instrument.replace('\t','').replace('\n', ''))] = 1
        return result

def get_spectrogram_data(audio_path):
    x, sr = librosa.load(audio_path)
    hop_length = 512
    n_mels = 128
    n_fft = 2048
    return librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)


create_datasets()
np.save(str(Path(__file__).parent) + "\\test_y", testing_y)
np.save(str(Path(__file__).parent) + "\\test_X", testing_X)

