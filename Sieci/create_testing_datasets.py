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

def create_spectograms():
    for filename in os.listdir(str(path.parent.parent) + f'\\IRMAS-TestingData-Part\\Part1\\'):
        audio_path = str(path.parent.parent) + f'\\IRMAS-TestingData-Part1\\Part1\\{filename}'
        if(filename[-3:] == 'wav'): 
            x, sr = librosa.load(audio_path)
            hop_length = 512
            n_mels = 128
            n_fft = 2048
            S = librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_DB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length)
            plt.savefig(str(path.parent.parent) + f'\\IRMAS-TestingDataPng\\{filename[:-3].replace(".", "")}.png', bbox_inches='tight', pad_inches=0)
            plt.clf()
        else:
            copyfile(audio_path, str(path.parent.parent) + f'\\IRMAS-TestingDataPng\\{filename[:-3].replace(".", "")}.txt')

def create_datasets():
    for filename in os.listdir(str(path.parent.parent) + f'\\IRMAS-TestingDataPng\\'):
        if(filename[-3:] == 'png'):
            testing_X.append(imageio.imread(str(Path(__file__).parent.parent) + f"\\IRMAS-TestingDataPng\\{filename}", as_gray=False, pilmode="RGB"))
        else:
            testing_y.append(get_categories(filename))
        print(filename)

def get_categories(filename):
    with open(str(Path(__file__).parent.parent) + f"\\IRMAS-TestingDataPng\\{filename}", 'r') as file:
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        data = file.readlines()
        for instrument in data:
            result[instruments_set.index(instrument.replace('\t','').replace('\n', ''))] = 1
        return result

#   UNCOMMENT TO CREATE TESTING SPECTOGRAMS
#   create_spectograms()

#   UNCOMMENT TO CREATE TESTING DATASETS
# create_datasets()
# np.save(str(Path(__file__).parent) + "\\test_y", testing_y)
# np.save(str(Path(__file__).parent) + "\\test_X", testing_X)

