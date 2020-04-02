import re
from enum import Enum
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imageio
from pathlib import Path

instruments_set = ('cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi')
genres = 'cel cla flu gac gel org pia sax tru vio voi'.split()
train_X = []
train_y = []

def get_categories(filename):
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    instruments = re.findall(r"\[(.+?)\]", filename)
    for instrument in instruments:
        try:
            result[instruments_set.index(instrument)] = 1
        except:
            print("Instrument dont match")
    return result

for g in genres:
    for filename in os.listdir(str(Path(__file__).parent.parent) + f'\\IRMAS-TrainingDataPng\\{g}'):
        train_y.append(get_categories(filename))
        train_X.append(imageio.imread(str(Path(__file__).parent.parent) + f"\\IRMAS-TrainingDataPng\\{g}\\{filename}", as_gray=False, pilmode="RGB"))
np.save(str(Path(__file__).parent) + "\\train_y", train_y)
np.save(str(Path(__file__).parent) + "\\train_X", train_X)









