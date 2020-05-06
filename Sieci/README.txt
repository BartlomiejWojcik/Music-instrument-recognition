Przygotowywanie danych:

-create_testing_datasets.py - skrypt do przygotowania danych testowych na podstawie spektoramów (z folderów: IRMAS-TestingData-Part1, IRMAS-TestingData-Part2, IRMAS-TestingData-Part3)
-create_training_datasets.py - skrypt do przygotowania danych treningowych na podstawie spektrogramów(z folderu IRMAS-TrainingData)

Powyższe skrypty generują odpowiednie pliki danych npy:

-test_X.npy
-test_y.npy
-train_X.npy
-train_y.npy

W przypadku danych dla modelu opartego na parametrach przetworzone dane znajdują się w plikach csv:

-param_data_train.csv
-param_data_test.csv-

Tworzenie modeli:

-network_spectrogram.ipynb - skrypt do utworzania modelu na podstawie spektrogramów
-network_parameters.ipynb - skrypt do utworzenia modelu na podstawie parametrów
-both_parameters.ipynb - skrypt do utworzenia modelu połączonego (spektorogramy + parametry)

Ocenianiem modeli zajmują się odpowiadające skrypty:

-evaluate_spectrogram.ipnyb
-evaluate_parameters.ipnyb
-evaluate_both.ipnyb