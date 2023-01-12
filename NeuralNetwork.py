import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
import time
import winsound
from datetime import datetime
from tensorflow.math import reduce_mean, abs, reduce_sum
#from RegressionSV import Performance
import tensorflow as tf
from decimal import Decimal

from tensorflow import keras
from tensorflow.keras import layers

from General_functions import ML_database ,simple_database, make_IAST_database_Wessel_version  

def beep():
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
    
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

def build_and_compile_model(no_mols):
    shape_input_layer = no_mols*9 +2
    model = keras.Sequential([
        keras.Input(shape=(shape_input_layer,)),
        layers.Dense(50, activation = 'sigmoid'),
        layers.Dense(100, activation = 'sigmoid'),
        layers.Dense(150, activation = 'sigmoid'),
        layers.Dense(200, activation = 'relu'),
        layers.Dense(150, activation = 'sigmoid'),
        layers.Dense(100, activation = 'sigmoid'),
        layers.Dense(no_mols)
    ])
    
    model.compile(loss='mean_absolute_error',
                  optimizer = tf.keras.optimizers.Adam(0.001))
    return model

def column_name_func(no_mols):
    column_list=[]
    for i in range(no_mols):
        column_list.append('frac'+str(i))
    for i in range(no_mols):
        column_list.extend(['index_Mol_'+str(i),'N.o._C_atoms_'+str(i),'N.o._Branches_'+str(i),'Location_branch1_'+str(i),'Length_branch1_'+str(i),'Location_branch2_'+str(i),'Length_branch2_'+str(i),'Location_branch3_'+str(i),'Length_branch3_'+str(i)])
    column_list.extend(['Pressure [Pa]','Temperature [K]'])
    return column_list
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

print(tf.__version__)


#import database of molecules representation
selfies_database = ML_database()
easy_database = simple_database()
no_mols = 4

#getting IAST combined with the molecule representation
x_data, y_data = make_IAST_database_Wessel_version(easy_database, no_mols)

columnname = column_name_func(no_mols)
dataset = pd.DataFrame(np.c_[y_data,x_data],columns = columnname)
#print(dataset.isna().sum()) #Check if there are any rows with empy values
dataset = dataset.dropna() #Delete the rows if they have emptyvalues

train_dataset = dataset.sample(frac=0.8, random_state=0) #Take 80% of the data to train
test_dataset = dataset.drop(train_dataset.index) #Take the rest of the data to test features, in this case we try to mol fractions wrt to the molecule geometry, the pressure and the temperature
train_features = train_dataset.copy()
test_features = test_dataset.copy()

#Make labels, it takes an extra loop since the pop function (which takes one column from a pandas DataFrame) can only take one column at a time
train_labels = pd.DataFrame()
test_labels = pd.DataFrame()
for i in range(no_mols):
    train_labels[columnname[i]] = train_features.pop(columnname[i])
    test_labels[columnname[i]] = test_features.pop(columnname[i])
print(f"NOW TRAINING FOR {no_mols} MOLECULES\nI've started at {datetime.now()}")

'''
dnn = build_and_compile_model(no_mols)


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

dnn.save('dnn_model_'+str(no_mols)+'_mols')
'''

#print(f"Ive started loading the model at {datetime.now()}")
#starttime = time.time()
dnn = keras.models.load_model('dnn_model_'+str(no_mols)+'_mols')
#print(f'Loading took {starttime-time.time()}[sec]')
#test_predictions = dnn.predict(test_features)

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

test_results = dnn.evaluate(test_features, test_labels, verbose=0)
'''



#https://www.youtube.com/watch?v=K03Uve6fgFM&ab_channel=CodingLikeMad

Performance("Neural Network" , no_mols, dnn , train_features.to_numpy(), test_features.to_numpy(), train_labels.to_numpy(), test_labels.to_numpy())
beep()