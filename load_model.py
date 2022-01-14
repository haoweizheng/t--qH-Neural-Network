import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras

from sklearn.metrics import confusion_matrix



y_vars = ['num_jets', 'mjj', 'y1_pt', 'y1_eta', 'y1_E', 'y2_pt', 'y2_eta', 'y2_E']
jet_vars = ['jet%d_pt', 'jet%d_eta', 'jet%d_E','jet%d_DLR1'] 
labels = []
for x in range(0,4):
    for var in jet_vars:
        labels.append(var %x)
labels = y_vars + labels

signal = pd.read_csv("../signal.csv", names = labels, sep = ",")
background = pd.read_csv("../bkgd.csv", names = labels, sep = ",")

signal_std = pd.read_csv("../signal_standardized.csv", names = labels, sep = ",")
background_std = pd.read_csv("../bkgd_standardized.csv", names = labels, sep = ",")


#seperate signals and backgrounds
signal["isSignal"] = 1
background["isSignal"] = 0

signal_std["isSignal"] = 1
background_std["isSignal"] = 0

dataset = pd.concat([signal, background])
dataset_std = pd.concat([signal_std, background_std])

dataset = dataset.sample(frac = 1)
dataset_std = dataset_std.sample(frac = 1)

X = dataset.drop("isSignal", axis = 1)
Y = dataset["isSignal"]

X_std = dataset_std.drop("isSignal", axis = 1)
Y_std = dataset_std["isSignal"]

model = keras.models.load_model('model')

model.summary()

Y_predicted_std = model.predict(X_std)
Y_predicted_1_std = [int(round(i[0])) for i in Y_predicted_std]

Y_predicted = model.predict(X)
Y_predicted_1 = [int(round(i[0])) for i in Y_predicted]

matrix = confusion_matrix(Y, Y_predicted_1, normalize = 'true')
matrix_std = confusion_matrix(Y_std, Y_predicted_1_std, normalize = 'true')

print("Normal: \n")
print(matrix)
print("Standardised: \n")
print(matrix_std)

