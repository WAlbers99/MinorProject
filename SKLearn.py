# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import time

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,classification_report,confusion_matrix
from math import sqrt
from sklearn.metrics import r2_score
from General_functions import ML_database ,simple_database, make_IAST_database_Wessel_version  
import tensorflow as tf
# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense

#making database of molecules representation
selfies_database = ML_database()
easy_database = simple_database()

#getting IAST combined with the molecule representaion
x_data, y_data = make_IAST_database_Wessel_version(easy_database,3)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size= 0.3)

print(x_train.shape); print(x_test.shape)
start = time.time()
model = Sequential()
model.add(Dense(50, input_dim=29, activation= "sigmoid"))
model.add(Dense(100, activation= "sigmoid"))
model.add(Dense(150, activation= "sigmoid"))
model.add(Dense(300, activation= "sigmoid"))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation= "relu"))
model.add(Dense(100, activation="sigmoid"))
model.add(Dense(3))


model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
model.fit(x_train, y_train, epochs=30)
end = time.time()
print("Model training took: ",end-start,"[sec]")
pred_train= model.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train)))


pred= model.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))