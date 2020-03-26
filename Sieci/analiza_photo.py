from keras.activations import relu, softmax
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation
import os
from PIL import Image
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

genres = 'cel cla flu gac gel org pia sax tru vio voi'.split()
X_train=np.zeros((480,640,132),np.uint8)
y_train = []
i = 0

# write specgrams to X_train and y_train list
for g in genres:
   for filename in os.listdir(f'C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/try/{g}'):
       y_train.append(g)
       X_train[:, :, i] = cv2.imread(f'C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/try2/{g}/{filename}', 0)-255 # ewentualnie można bez tego -255
       i += 1

y_train = np.array(y_train)

X_train = X_train.reshape(132,480,640,1)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_train = to_categorical(y_train)

#create model
model = Sequential()

#add model layers
model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape=(1100,1100,1)))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(11, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, epochs=20, batch_size=3)


