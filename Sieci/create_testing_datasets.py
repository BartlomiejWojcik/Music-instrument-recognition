import librosa, os, sys, warnings, imageio
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
from pathlib import Path
from shutil import copyfile


path = Path(os.path.abspath(''))

warnings.filterwarnings('ignore')

instruments_set = ('cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi')
#genres = 'cel cla flu gac gel org pia sax tru vio voi'.split()
testing_X = []
testing_y = []

# TODO: Change audio duration to 3s 

def create_datasets():
    for filename in os.listdir(str(path) + f'\\IRMAS-TestingData-Part1\\Part1\\'):
        print(filename)
        audio_path = str(path) + f'\\IRMAS-TestingData-Part1\\Part1\\{filename}'
        if(filename[-3:] == 'wav'): 
            testing_X.append(get_spectrogram_data(audio_path))
        else:
            testing_y.append(get_categories(audio_path))
    for filename in os.listdir(str(path) + f'\\IRMAS-TestingData-Part2\\IRTestingData-Part2\\'):
        print(filename)
        audio_path = str(path) + f'\\IRMAS-TestingData-Part2\\IRTestingData-Part2\\{filename}'
        if(filename[-3:] == 'wav'): 
            testing_X.append(get_spectrogram_data(audio_path))
        else:
            testing_y.append(get_categories(audio_path))
    for filename in os.listdir(str(path) + f'\\IRMAS-TestingData-Part3\\Part3\\'):
        print(filename)
        audio_path = str(path) + f'\\IRMAS-TestingData-Part3\\Part3\\{filename}'
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
    x, sr = librosa.load(audio_path, duration=3)
    return librosa.feature.melspectrogram(x, sr=sr, n_fft=2048, hop_length=512, n_mels=128)


create_datasets()
np.save(str(path) + "\\test_y", testing_y)
np.save(str(path) + "\\test_X", testing_X)

