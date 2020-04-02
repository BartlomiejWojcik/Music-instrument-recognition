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
import librosa
import re

path = Path(__file__)

warnings.filterwarnings('ignore')

instruments_set = ('cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi')
genres = 'cel cla flu gac gel org pia sax tru vio voi'.split()
testing_X = []
testing_y = []

def create_datasets():
    for g in genres:
        print(g)
        for filename in os.listdir(str(Path(__file__).parent.parent) + f'\\IRMAS-TrainingData\\{g}'):
            print(filename)
            audio_path = str(path.parent.parent) + f'\\IRMAS-TrainingData\\{g}\\{filename}'
            if(filename[-3:] == 'wav'): 
                x, sr = librosa.load(audio_path)
                hop_length = 512
                n_mels = 128
                n_fft = 2048
                testing_X.append(librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels))
                testing_y.append(get_categories(filename))
        
def get_categories(filename):
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    instruments = re.findall(r"\[(.+?)\]", filename)
    for instrument in instruments:
        try:
            result[instruments_set.index(instrument)] = 1
        except:
            print(filename)
    return result

create_datasets()

np.save(str(Path(__file__).parent) + "\\train_y", testing_y)
np.save(str(Path(__file__).parent) + "\\train_X", testing_X)

