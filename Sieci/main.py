from analiza import *

#create_specgrams()
#create_specgrams_patki()
#createCsvData()
X, y = createDataToNetwork()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#create model
model = Sequential()

#add model layers
model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(11, activation='softmax'))

#compile model using accuracy to measure model performance
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=3)

print(predictWavFile('C:/Users/Lenovo/Desktop/Magisterka/Sieci neuronowe/IRMAS-TrainingData/cel/[cel][cla]0001__1.wav', model))
print(model.predict(X_test[:1]))
print(y_test[:1])