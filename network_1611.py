import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

#define stuff
e = int(input("How many epochs do you want to run? \n"))
b = int(input("What is the batch size? \n"))
fraction = float(input("What is the fraction of the background used for training? \n"))
l_rate = float(input("What is the optimiser learning rate? \n"))

#make labels
y_vars = ['num_jets', 'mjj', 'y1_pt', 'y1_eta', 'y1_E', 'y2_pt', 'y2_eta', 'y2_E']
jet_vars = ['jet%d_pt', 'jet%d_eta', 'jet%d_E','jet%d_DLR1'] 
labels = []
for x in range(0,4):
    for var in jet_vars:
        labels.append(var %x)
labels = y_vars + labels


#import and shuffle signal and background
signal = pd.read_csv("../signal_standardized.csv", names = labels, sep = ",")
background = pd.read_csv("../bkgd_standardized.csv", names = labels, sep = ",")
signal = signal.sample(frac = 1)
background = background.sample(frac = 1)

#seperate signals and backgrounds
signal["isSignal"] = 1
background["isSignal"] = 0

#seperate fit and test datasets
signal_fit, signal_test = train_test_split(signal, test_size = 0.2)
background_fit, background_test = train_test_split(background, test_size = 0.2)
background_fit = background_fit.sample(frac = fraction)
len_signal_test = int(len(signal_test))

#useful for later
len_background_test = int(len(background_test))

#print stuff for clarity
print("Number of siganls = " + str(len(signal_fit)) + "\n")
print("Number of backgrounds = " + str(len(background_fit)) + "\n")


#finally make the datasets
dataset_fit = pd.concat([signal_fit, background_fit])
dataset_test = pd.concat([signal_test, background_test])
input_dimension = len(dataset_fit.columns) - 1
dataset_fit = dataset_fit.sample(frac = 1)
dataset_test = dataset_test.sample(frac = 1)
print("Input dimenstion is: "+ str(input_dimension) + ". \n")


#seperate into x and y
X_fit = dataset_fit.drop("isSignal", axis = 1)
Y_fit = dataset_fit["isSignal"]

X_test = dataset_test.drop("isSignal", axis = 1)
Y_test = dataset_test["isSignal"]

#define the early stopping callback and optimiser
early_stopping = EarlyStopping(patience = 4)
opt = tf.keras.optimizers.Adam(learning_rate=l_rate)

#define the neural network model
model = Sequential()
model.add(Dense(30, input_dim=input_dimension, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compile the model and fit it
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.AUC(curve = "PR")])
fit_history = model.fit(X_fit, Y_fit, epochs=e, batch_size=b, validation_split = 0.2, callbacks = [early_stopping])

#predict with model
Y_predicted = model.predict(X_test)
Y_predicted_1 = [int(round(i[0])) for i in Y_predicted]
Y_test = np.array(Y_test)
Y_fit_predicted = model.predict(X_fit)

dataset_test["Y_predicted"] = Y_predicted

#find confusion matrix
matrix = confusion_matrix(Y_test, Y_predicted_1, normalize = 'true')
roc_x, roc_y, roc_threshold = roc_curve(Y_test, Y_predicted)
roc_f_x, roc_f_y, roc_f_threshold = roc_curve(Y_fit, Y_fit_predicted)
area = auc(roc_x, roc_y)
area_f = auc(roc_f_x, roc_f_y)

print("Area under Curve: " + str(area) + "\n")
print("CoNfUsIoN MaTrIx: \n")
print(matrix)

model.save('model')

dot_img_file = 'model/model.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)



hist = pd.DataFrame(fit_history.history)
hist.to_csv("model/history.csv") 

roc_c = {'fpr':roc_x, 'tpr':roc_y}
roc_curve = pd.DataFrame(data = roc_c)
roc_curve.to_csv("model/roc.csv")

roc_f = {'fpr':roc_f_x, 'tpr':roc_f_y}
roc_curve_f = pd.DataFrame(data = roc_f)
roc_curve_f.to_csv("model/roc_f.csv")

dataset_fit.to_csv("model/fit_data.csv")
dataset_test.to_csv("model/test_data.csv")


model.summary()
