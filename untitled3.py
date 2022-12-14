import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

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

def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation = 'relu'),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(1)
    ])
    
    model.compile(loss='mean_absolute_error',
                  optimizer = tf.keras.optimizers.Adam(0.001))
    return model

print(tf.__version__)


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
#Import raw data
raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()#copy raw data into dataset variable
#print(dataset.tail()) #print the last few rows of the dataset


#print(dataset.isna().sum())#print the columns with empty values

dataset = dataset.dropna()#drop the rows from the dataset

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2:'Europe', 3:'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
train_dataset.describe().transpose()

#Seperate the target value from the features, in this case we try to predict the mileage or Miles per Gallon MPG
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

#Normalization
#print(train_dataset.describe().transpose()[['mean','std']])
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

#print(normalizer.mean.numpy())

first = np.array(train_features[:1])

horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)
test_results = {}

'''
#HORSEPOWER MODEL!!!!
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

horsepower_model.summary()

horsepower_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1),
    loss='mean_absolute_error')

history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    verbose=0,
    validation_split = 0.2)
hist = pd.DataFrame(history.history)
hist['epochs'] = history.epoch
print(hist.tail())

plot_loss(history)

test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

plot_horsepower(x, y)


#LINEAR MODEL!!!!
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

linear_history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2)

plot_loss(linear_history)
test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)
'''

dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
dnn_horsepower_model.summary()

history = dnn_horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    validation_split=0.2,
    verbose = 0,
    epochs = 100)

plot_loss(history)

x = tf.linspace(0.0,250,251)
y = dnn_horsepower_model.predict(x)

plot_horsepower(x,y)
