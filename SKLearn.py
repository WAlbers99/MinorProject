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

def model_build_compile(fold_no,mol0,mol1,mol2,mol3,mol4,mol5,act0,act1,act2,act3,act4,act5,act6):
    shape_input_layer = no_mols*9+2
    
    model = Sequential()
    model.add(Dense(mol0, input_dim=shape_input_layer, activation= act0))
    model.add(Dense(mol1, activation= act1))
    model.add(Dense(mol2, activation= act2))
    model.add(Dense(mol3, activation=act3))
    model.add(Dense(mol4, activation= act4))
    model.add(Dense(mol5, activation=act5))
    model.add(Dense(no_mols, activation=act6))
    
    model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["accuracy"])
    print('-----------------------------------------------')
    print(f'Training for fold {fold_no}')
    start = time.time()
    model.fit(x_train, y_train, epochs=15)
    end = time.time()
    print("Model training took: ",end-start,"[sec]")
    scores = model.evaluate(x_test,y_test,verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1]*100)
    loss_per_fold.append(scores[0])
    
    fold_no = fold_no + 1
    return model

def cross_validation(model, x_train, y_train, splits=5):
    kf = KFold(n_splits = splits)

def save_model(model, no_mols):
    dump(model, 'model_'+str(no_mols))
def load_model(no_mols):
    return load('model_'+str(no_mols))
no_mols=2

#making database of molecules representation
selfies_database = ML_database()
easy_database = simple_database()
#getting IAST combined with the molecule representaion
x_data, y_data = make_IAST_database_Wessel_version(easy_database,no_mols)
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size= 0.2)

acc_per_fold = []
loss_per_fold = []

kfold = KFold(n_splits=5, shuffle=True)

fold_no=1
''' 
for train, test in kfold.split(x_data,y_data):
'''

#save_model(NN, no_mols)
#NN = load_model(no_mols)
'''
print('--------------------------------')
print('Score per fold')
for i in range(0,len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

'''


#model = load('model_'+str(no_mols))
act_lst = ['sigmoid','relu']
no_nodes = [10,50,100,150,200]
for node_0 in no_nodes:
    for node_1 in no_nodes:
        for node_2 in no_nodes:
            for node_3 in no_nodes:
                for node_4 in no_nodes:
                    for node_5 in no_nodes:
                        for act0 in act_lst:
                            for act1 in act_lst:
                                for act2 in act_lst:
                                    for act3 in act_lst:
                                        for act3 in act_lst:
                                            for act4 in act_lst:
                                                for act5 in act_lst:
                                                    for act6 in act_lst:
                                                        NN = model_build_compile(fold_no,node_0,node_1,node_2,node_3,node_4,node_5,act0,act1,act2,act3,act4,act5,act6)
                                                        name = "NN:"+str(node_0)+str(node_1)+str(node_2)+str(node_3)+str(node_4)+str(node_5)+str(act0)+str(act1)+str(act2)+str(act3)+str(act4)+str(act5)+str(act6)
                                                        Performance(name , no_mols, NN , x_train, x_test, y_train, y_test)
                                                    
                                        


#beep()