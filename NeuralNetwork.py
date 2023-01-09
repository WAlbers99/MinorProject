import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import winsound

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from General_functions import ML_database ,simple_database, make_IAST_database_Wessel_version  
def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0,10])
    plt.xlabel('Epochs')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

def plot_horsepower(x,y):
    plt.figure()
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color = 'k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()

def build_and_compile_model(no_atoms):
    shape_input_layer = no_atoms*9 +2
    model = keras.Sequential([
        keras.Input(shape=(shape_input_layer,)),
        layers.Dense(50, activation = 'sigmoid'),
        layers.Dense(100, activation = 'sigmoid'),
        layers.Dense(150, activation = 'sigmoid'),
        layers.Dense(200, activation = 'relu'),
        layers.Dense(150, activation = 'sigmoid'),
        layers.Dense(100, activation = 'sigmoid'),
        layers.Dense(no_atoms)
    ])
    
    model.compile(loss='mean_absolute_error',
                  optimizer = tf.keras.optimizers.Adam(0.001))
    return model

def column_name_func(no_atoms):
    column_list=[]
    for i in range(no_atoms):
        column_list.append('frac'+str(i))
    for i in range(no_atoms):
        column_list.extend(['index_Mol_'+str(i),'N.o._C_atoms_'+str(i),'N.o._Branches_'+str(i),'Location_branch1_'+str(i),'Length_branch1_'+str(i),'Location_branch2_'+str(i),'Length_branch2_'+str(i),'Location_branch3_'+str(i),'Length_branch3_'+str(i)])
    column_list.extend(['Pressure [Pa]','Temperature [K]'])
    return column_list
print(tf.__version__)


#import database of molecules representation
selfies_database = ML_database()
easy_database = simple_database()
no_atoms = 4

#getting IAST combined with the molecule representation
x_data, y_data = make_IAST_database_Wessel_version(easy_database, no_atoms)

columnname = column_name_func(no_atoms)
dataset = pd.DataFrame(np.c_[y_data,x_data],columns = columnname)
print(dataset.isna().sum()) #Check if there are any rows with empy values
dataset = dataset.dropna() #Delete the rows if they have emptyvalues

train_dataset = dataset.sample(frac=0.8, random_state=0) #Take 80% of the data to train
test_dataset = dataset.drop(train_dataset.index) #Take the rest of the data to test features, in this case we try to mol fractions wrt to the molecule geometry, the pressure and the temperature
train_features = train_dataset.copy()
test_features = test_dataset.copy()

#Make labels, it takes an extra loop since the pop function (which takes one column from a pandas DataFrame) can only take one column at a time
train_labels = pd.DataFrame()
test_labels = pd.DataFrame()
for i in range(no_atoms):
    train_labels[columnname[i]] = train_features.pop(columnname[i])
    test_labels[columnname[i]] = test_features.pop(columnname[i])

dnn = build_and_compile_model(no_atoms)
start_time_training = time.time()
history = dnn.fit(
    train_features,
    train_labels,
    validation_split = 0.2,
    verbose = 0,
    epochs = 100)
end_time_training = time.time()
print("Training took: ",end_time_training-start_time_training)

plot_loss(history)

dnn.save('dnn_model')

test_predictions = dnn.predict(test_features)
'''
a = plt.axes(aspect='equal')
plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Fractions]')
plt.ylabel('Predictions [Fractions]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

plt.figure()
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [Fractions]')
_ = plt.ylabel('Count')
plt.show()
'''
test_results = dnn.evaluate(test_features, test_labels, verbose=0)
#test_results['dnn'] = dnn.evaluate(test_features, test_labels, verbose=0)

#Make a sound when finished
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
#https://www.youtube.com/watch?v=K03Uve6fgFM&ab_channel=CodingLikeMad