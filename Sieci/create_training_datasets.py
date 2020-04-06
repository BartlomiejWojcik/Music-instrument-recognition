import librosa, os, sys, warnings, imageio, librosa, re, imageio, librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
from pathlib import Path
from shutil import copyfile

path = Path(os.path.abspath(''))

warnings.filterwarnings('ignore')

instruments_set = ('cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi')
genres = 'cel cla flu gac gel org pia sax tru vio voi'.split()
testing_X = []
testing_y = []

def create_datasets():
    for g in genres:
        print(g)
        for filename in os.listdir(str(path) + f'\\IRMAS-TrainingData\\{g}'):
            print(filename)
            audio_path = str(path) + f'\\IRMAS-TrainingData\\{g}\\{filename}'
            testing_X.append(get_spectrogram_data(audio_path))
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

def get_spectrogram_data(audio_path):
    x, sr = librosa.load(audio_path, duration=3)
    return librosa.feature.melspectrogram(x, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

create_datasets()
np.save(str(path) + "\\train_y", testing_y)
np.save(str(path) + "\\train_X", testing_X)

