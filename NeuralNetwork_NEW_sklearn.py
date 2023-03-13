# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPRegressor
import time

# Import necessary modules
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error,classification_report,confusion_matrix
from math import sqrt
from sklearn.metrics import r2_score
from General_functions import ML_database ,simple_database, make_IAST_database_Wessel_version  
import tensorflow as tf
# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from decimal import Decimal
from joblib import dump, load
import winsound
from sklearn.model_selection import cross_validate


def beep():
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
    
def Performance(name_model, amount_mols, rf_model, x_train, x_test, y_train, y_test):
    """
    Performance function, to let it work properly, please let the inputs be of 
    the following format:
            
         -   name_model: enter the name of the model as a string, will be used in
             titles of plots, for consistency hold the format like for example:
             "Decision Tree" and "Neural Network"
         -   amount_mols: the amount of molecules in the mixture used to calculate the 
             loadings.
         -   rf_model: the variable where your model is stored, which is already being loaded in
             so first load the joblib model, store it in a variable and put the variable here.
         -   x_train: the part of the data used to train the model
         -   x_test: the part of the data to be used to test, which is not the same as the 
             training data!
         -   y_train: the known output of the x_train data
         -   y_test: the known output of the y_train data
    
    This function will make predictions of the x_test data, and times how long 
    it takes. Then it will calculate the absolute and relative error, the score
    of the model which is a build-in function of sklearn, which is described as
    the following:
        ######################################################################
        Return the coefficient of determination R^2 of the prediction.
    
        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a
            precomputed kernel matrix or a list of generic objects instead,
            shape = (n_samples, n_samples_fitted),
            where n_samples_fitted is the number of
            samples used in the fitting for the estimator.
        
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.
        ######################################################################

    Finally will this function give you the desired plots and will store them 
    as a pdf file.
    """
    pred_time = 0
    for i in range(1,10):
        print(i)
        start_time = time.time()
        y_pred=rf_model.predict(x_test) 
        end_time = time.time()
        pred_time_temp = end_time-start_time
        pred_time = (pred_time + pred_time_temp)/i
        print(pred_time)
        
        
    abs_err = np.abs(y_pred-y_test)
    rel_err = abs_err/y_test
    
    nanIndex = np.isnan(rel_err)
    rel_err = rel_err[~nanIndex] #to remove nan's
    infIndex = np.isinf(rel_err)
    rel_err = rel_err[~infIndex] #to remove zero's
    
    abs_err = abs_err[~nanIndex] #to remove nan's
    abs_err = abs_err[~infIndex] #to remove zero's
    
    rel_err = rel_err.flatten()
    abs_err = abs_err.flatten()
    
    mean_rel_err = np.mean(rel_err)
    mean_abs_err = np.mean(abs_err)
    
    new_name = ""
    for i in name_model.split(" "):
        new_name += i
    
    plt.figure()
    plt.title(f"Relative error {name_model}, {amount_mols} molecules mixture\nMean relative error = {'%.0e' %Decimal(mean_rel_err)}")
    plt.scatter(range(len(rel_err)), rel_err, s=4,label="Relative error point i")
    plt.hlines(mean_rel_err, xmin = 0, xmax = len(rel_err), color="red", label="Mean relative error")
    plt.yscale("log")
    plt.xlabel("Index of datapoint in array")
    plt.ylabel("Relative error of predicted point wrt to known point")
    plt.legend()
    plt.savefig(f"{new_name}_{amount_mols}molsmix_RelErrPlot")
    plt.show()
    
    plt.figure()
    plt.title(f"Absolute error {name_model}, {amount_mols} molecules mixture\nMean absolute error = {'%.0e' %Decimal(mean_abs_err)}")
    plt.scatter(range(len(abs_err)), abs_err, s=4, label="Absolute error point i")
    plt.hlines(mean_abs_err, xmin = 0, xmax = len(abs_err), color="red", label="Mean absolute error")
    plt.yscale("log")
    plt.xlabel("Index of datapoint in array")
    plt.ylabel("Absolute error of predicted point wrt to known point")
    plt.legend()
    plt.savefig(f"{new_name}_{amount_mols}molsmix_AbsErrPlot")
    plt.show()
    
    plt.figure()
    plt.title(f"Performance {name_model}, {amount_mols} molecules mixture")
    plt.scatter(y_test, y_pred, s=10)
    plt.xlabel("calculated loading by IAST (mol/kg)")
    plt.ylabel(f"Predicted loading {name_model} (mol/kg)")
    plt.savefig(f"{new_name}_{amount_mols}molsmix_PlotCompPredTrue")
    plt.show()
    
    print(f"\nMean relative error = {mean_rel_err}")
    print("Formula relative error: np.abs(y_pred-y_test)/y_test\n")
    print(f"Mean absolute error = {mean_abs_err}")
    print("Formula absolute error: np.abs(y_pred-y_test)\n")

    #print(f"Score model (based on test data) = {rf_model.score(x_test,y_test)}")
    #print(f"Score model (based on train data) = {rf_model.score(x_train,y_train)}\n")
   
    print(f"Total time to predict {len(y_pred)} amount of molmixes (mixture of {amount_mols} mols): {'%.2e' %Decimal(pred_time)}")
    print(f"Time to predict loading 1 molmix (mixture of {amount_mols} mols): {'%.2e' %Decimal((pred_time)/len(y_pred))}")

    f = open(f"{new_name}_{amount_mols}molsmix_performance.txt","w+")
    f.write(f"\nMean relative error = {mean_rel_err}\n")
    f.write("Formula relative error: np.abs(y_pred-y_test)/y_test\n")
    f.write(f"Mean absolute error = {mean_abs_err}\n")
    f.write("Formula absolute error: np.abs(y_pred-y_test)\n")

    #f.write(f"Score model (based on test data) = {rf_model.score(x_test,y_test)}\n")
    #f.write(f"Score model (based on train data) = {rf_model.score(x_train,y_train)}\n")
   
    f.write(f"Total time to predict {len(y_pred)} amount of molmixes (mixture of {amount_mols} mols): {'%.2e' %Decimal(pred_time)}\n")
    f.write(f"Time to predict loading 1 molmix (mixture of {amount_mols} mols): {'%.2e' %Decimal((pred_time)/len(y_pred))}\n")
    f.close()
    return 0;

def model_build_compile(no_mols):
    #Determine the shape/length of the input vector once the model is compiled it can't be changed or you have to compile a new model
    shape_input_layer = no_mols*9+2
    
    model = Sequential()#Determine type of model
    model.add(Dense(50, input_dim=shape_input_layer, activation= "relu"))#For each layer determine it's activation function and amount of nodes
    model.add(Dense(100, activation= "relu"))
    model.add(Dense(150, activation= "relu"))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(150, activation= "relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(no_mols, activation="relu"))
    #Now compile the model and determine what it will try to minimize, here it is set to mean_square_error
    #Adam is a learning optimizer specifically build for neural networks
    #Metric functions are similar to loss functions, except that the results from evaluating a metric are not used when training the model. Note that you may use any loss function as a metric.
    model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["accuracy"])
    print('-----------------------------------------------')
    print(f'Training')
    start = time.time()
    #fit is used to train the model, epochs is the number of iterations it will use the dataset
    model.fit(x_train, y_train, epochs=1)
    end = time.time()
    print("Model training took: ",end-start,"[sec]")
    scores = model.evaluate(x_test,y_test,verbose=0)
    print(f'Score: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    #acc_per_fold.append(scores[1]*100)
    #loss_per_fold.append(scores[0])
    

    return model

def save_model(model, no_mols):
    dump(model, 'model_'+str(no_mols))
def load_model(no_mols):
    return load('model_'+str(no_mols))

#How many molecules in this combination
no_mols=3

#making database of molecules representation
#selfies_database = ML_database()
easy_database = simple_database()
#getting IAST combined with the molecule representaion
x_data, y_data = make_IAST_database_Wessel_version(easy_database,no_mols)#Make database
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size= 0.2)#Split into test and train


NN = model_build_compile(no_mols)#Actually build the model


#save_model(NN, no_mols)
#NN = load_model(no_mols) #Load model instead of training






Performance("NeuralNetwork" , no_mols, NN , x_train, x_test, y_train, y_test)
                                                    
                                        


#beep()