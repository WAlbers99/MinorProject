# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 21:21:43 2022

@author: Wessel Albers
"""

import selfies as sf
import numpy as np
import os
import pandas as pd
import glob

def ML_database():
    
    
    "Creating the smiles and name arrays"
    molecule_names = ['C7',"2mC6",'3mC6','22mC5',"23mC5",'24mC5',"33mC5",  "3eC5",   "223mC4"]
    smiles_dataset = ["CCCCCCC", "CCCCC(C)C" ,"CCCC(C)CC" , "CCCC(C)(C)C" ,"CCC(C)C(C)C" , "CC(C)CC(C)C" ,"CCC(C)(C)CC","CCC(CC)CC" ,"CC(C)C(C)(C)C"]
    selfies_dataset = list(map(sf.encoder, smiles_dataset)) #transforming to selfies
    
    
    max_len = max(sf.len_selfies(s) for s in selfies_dataset)
    symbols = sf.get_alphabet_from_selfies(selfies_dataset) # creating symbols for each character that is in the database
    symbols.add("[nop]") # this is an padding symbol, otherwise it does not work
    
    vocab_stoi = {symbol: idx for idx, symbol in enumerate(symbols)} #giving idx to each symbol
    
    "creating dictionary for storage"
    molecular_database = {} 
    
    for first, name in zip(selfies_dataset,molecule_names):
        'Creating one_hot encoding'
        one_hot = np.array(sf.selfies_to_encoding(first, vocab_stoi, pad_to_len =max_len)[1])
        one_hot = one_hot.reshape(one_hot.shape[1]*one_hot.shape[0])#rescaling into a vector
        
        "Adding vector to the dictionary with name"
        molecular_database[name] = one_hot

    return molecular_database

def simple_database():
    database = {}
    # array [c atomen, hoeveel braches, hoeveel c atomen in branches]
    database["C7"] = np.array([7,0,0])
    database["2mC6"] = np.array([7,1,1])
    database["3mC6"] = np.array([7,1,1])
    database['22mC5'] = np.array([7,2,1])
    database["23mC5"] = np.array([7,2,1])
    database['24mC5'] = np.array([7,2,1])
    database['33mC5'] = np.array([7,2,1])
    database["3eC5"] = np.array([7,1,2])
    database['223mC4'] = np.array([7,3,1])
    
    return database


def data_gathering(path_to_output):
    data = {}
    outputmaps = os.listdir(path_to_output)
    for outputmap in outputmaps:
        mappath = path_to_output + "/" + str(outputmap)
        if os.path.isdir(mappath):
            files = os.listdir(mappath)
            for file in files:
                try:
                    paths =  mappath + "/" + str(file)
                    label = file.split("out")[0]
                   #print(label)
                    df = pd.read_table(paths, delimiter = ",")
                    #df = df.set_index("pressure")
                    data[label] = df.drop(["_","muc", "muc_err"], axis = 1)
                    #print(data)
                except:
                    print("ERROR !!!, please check " + file + " \n")
    return data

def remove_pressures_RASPA(columns):
    path_to_out='MachineLearning/Outputs_RASPA'
    paths = glob.glob(path_to_out + "/*.txt")
    for file in paths:
        data=np.genfromtxt(file,delimiter=',',usecols=columns,skip_header=1)
        data=np.delete(data,obj=np.where(data[:,0]>1e8),axis=0)
        temp=int( file.split('/')[-1].split("out")[0][-3:] )
        data=np.insert(data,obj=1,axis=1,values=temp*np.ones(np.shape(data)[0]))
        np.savetxt(file, data,header='pressure,temperature,molkg',delimiter=',')

def remove_pressures_IAST():
    path_to_out='MachineLearning/Outputs_IAST'
    paths = glob.glob(path_to_out + "/*.txt")
    for file in paths:
        data=np.genfromtxt(file,delimiter='    ',skip_header=1,dtype=float)
        data=np.delete(data,obj=np.where(data[:,0]>1e8),axis=0)
        length=np.ones(np.shape(data)[0])
        file_split=file.split('-')
        
        temp=int(file_split[1])
        f1=float(file_split[2][:3])
        f2=float(file_split[-1][:3])
        
        data=np.insert(data,obj=1,axis=1,values=temp*length)
        data=np.insert(data,obj=2,axis=1,values=f1*length)
        data=np.insert(data,obj=3,axis=1,values=f2*length)
        np.savetxt(file, data,header='pressure,temperature,f1,f2,molkg1,molkg2',delimiter=',')
        
path_RASPA=glob.glob('MachineLearning/Outputs_RASPA/*.txt')

chemstructure=ML_database()
data_RASPA=[]
for file in path_RASPA:
    
    molecule = file.split('/')[-1].split('-')[0]

    data = np.loadtxt(file,skiprows=1,delimiter=',',usecols=(0,1,-1))  
    selfie=np.repeat(chemstructure[molecule], data.shape[0]).reshape(52,data.shape[0]).T
    data=np.hstack((selfie,data))
    data_RASPA.append(data)