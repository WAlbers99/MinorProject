from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split  
import os
import pandas as pd
# from General_functions import ML_database, make_IAST_database_ver3
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

import selfies as sf
import numpy as np
import os
import pandas as pd
import glob
import sys

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

def RASPA_database(columns,chemstructure=ML_database()):
    path_to_out='MachineLearning/Outputs_RASPA'
    paths = glob.glob(path_to_out + "/*.txt")
    database=np.array(len(paths))
    for i,file in enumerate(paths):
        molecule = file.split('/')[-1].split('-')[0]
        selfie=chemstructure[molecule]
        data=np.genfromtxt(file,delimiter=',',usecols=columns,skip_header=1)
        
        #Removing pressures that are too high
        data=np.delete(data,obj=np.where(data[:,0]>1e8),axis=0)
        
        #adding temperature
        temp=int( file.split('/')[-1].split("out")[0][-3:] )
        data=np.insert(data,obj=1,axis=1,values=temp*np.ones(np.shape(data)[0]))
        
        #adding selfie
        data=np.insert(data,obj=0,axis=0,values=selfie)
        database[i]=data
    return database

def IAST_database():
    path_to_out='MachineLearning/Outputs_IAST'
    paths = glob.glob(path_to_out + "/*.txt")
    
    for file in paths:
        #Removing pressures that are too high
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

def make_training_database(chemstructure=ML_database()):
    path_RASPA=glob.glob('MachineLearning/Outputs_RASPA/*.txt')
    path_IAST=glob.glob('MachineLearning/Outputs_IAST/*.txt')
    
    for i, file in enumerate(path_RASPA):
        file.replace("\\", "/")
    
    data_RASPA=[]
    data_IAST=[]
    
    for file in path_RASPA:
        file = file.replace("\\", "/") #comment this line if you use linux
        molecule = file.split('/')[-1].split('-')[0]

        data = np.loadtxt(file,skiprows=1,delimiter=',',usecols=(0,1,-1))  
        selfie=np.repeat(chemstructure[molecule], data.shape[0]).reshape(52,data.shape[0]).T
        data=np.hstack((selfie,data))
        data_RASPA.append(data)
        
    for file in path_IAST:
        file = file.replace("\\", "/") #comment this line if you use linux
        
        m1=file.split('/')[-1].split('-')[0]
        m2=file.split('/')[-1].split('-')[2][5:]
        
        f1=float( file.split('/')[-1].split('-')[2][:4] )
        f2=1-f1
        data=np.loadtxt(file,delimiter=',',skiprows=1,usecols=(0,1,-2,-1))
        try:
            selfie1=np.repeat(chemstructure[m1], data.shape[0]).reshape(52,data.shape[0]).T
            selfie2=np.repeat(chemstructure[m2], data.shape[0]).reshape(52,data.shape[0]).T
        except KeyError:
            selfie1=np.repeat(chemstructure['22mC5'], data.shape[0]).reshape(52,data.shape[0]).T
            selfie2=np.repeat(chemstructure[m2], data.shape[0]).reshape(52,data.shape[0]).T
        selfie=f1*selfie1+f2*selfie2
        data=np.hstack((selfie,data))
        data_IAST.append(data)
    return np.vstack(data_RASPA),np.vstack(data_IAST)

def make_training_database_ver2(max_amount_mols, chemstructure=ML_database()):
    # path_RASPA=glob.glob('MachineLearning/Outputs_RASPA/*.txt')
    path_IAST=glob.glob('IAST-segregated/automated_output/*')
    
    # data_RASPA=[]
    data_IAST=[]
    # print(path_IAST)
    # sys.exit(0)
    
    # for file in path_RASPA:
    #     file = file.replace("\\", "/") #comment this line if you use linux
    #     molecule = file.split('/')[-1].split('-')[0]

    #     data = np.loadtxt(file,skiprows=1,delimiter=',',usecols=(0,1,-1))  
    #     selfie=np.repeat(chemstructure[molecule], data.shape[0]).reshape(52,data.shape[0]).T
    #     data=np.hstack((selfie,data))
    #     data_RASPA.append(data)
    
    for path_amount in path_IAST:
        path_amount = path_amount.replace("\\", "/")
        # print(path_amount)
        amount_mols = int(path_amount.split('/')[-1].split('_')[0])
        # print(amount_mols)
        if amount_mols>max_amount_mols:
            break
        path_temps = glob.glob(path_amount + "/*")
        for folder in path_temps:
            folder = folder.replace("\\", "/") #comment this line if you use linux
            # print(folder)
            # print(folder.split('/')[-1].split("K")[0])
            temp = float(folder.split('/')[-1].split("K")[0])
            # print(folder + "/*.txt")
            molc_folder = glob.glob(folder + "/*")
            
            # print(molc_folder)
            for mols_mix in molc_folder:
                mols_mix = mols_mix.replace("\\", "/")
                # print(mols_mix)
                # print(type(amount_mols))
                mols = mols_mix.split('/')[-1].split('-')
                # print(molc_folder)
                path_fracs = glob.glob(mols_mix + "/*.txt")
                for file in path_fracs:
                    file = file.replace("\\", "/")
                    fracs = file.split('/')[-1].split('.txt')[0].split("-")
                    # print(fracs)
                    fracs = np.array(fracs, dtype=float)
                    # print(path_fracs)
                    # print(fracs)
                    try:
                        data=np.loadtxt(file,delimiter='   ',skiprows =1, usecols=(range(1,amount_mols+2)))#Pressure, loading m1, loading m2,..., loading mi
                    except:
                        print(f"failed to load file: {file}")
                        print(mols_mix)
                        break
                    if len(mols)<max_amount_mols:
                        # print(data.shape)
                        # print(np.zeros((max_amount_mols - len(mols), len(data))).shape)
                        data = np.append(data, np.zeros((len(data),max_amount_mols - len(mols))), axis =1)
                    
                    
                    # print(data.shape)
                    selfie = 0
                    for i, mol in enumerate(mols):
                        selfie += np.repeat(chemstructure[mol], data.shape[0]).reshape(52,data.shape[0]).T
                    # print(selfie.shape)
                    # print(data.shape)
                    temp_arr = np.full((len(data), 1), temp) 
                    frac_arr = np.full((len(data), 1), fracs)
                    data=np.hstack((selfie,frac_arr,temp_arr, data))
                    data_IAST.append(data)
    return np.vstack(data_IAST)


def make_training_database_ver3(wanted_amount_mols, chemstructure=ML_database()):
    # path_RASPA=glob.glob('MachineLearning/Outputs_RASPA/*.txt')
    path_IAST=glob.glob('IAST-segregated/automated_output/*')
    
    # data_RASPA=[]
    data_IAST=[]
    # print(path_IAST)
    # sys.exit(0)
    
    # for file in path_RASPA:
    #     file = file.replace("\\", "/") #comment this line if you use linux
    #     molecule = file.split('/')[-1].split('-')[0]

    #     data = np.loadtxt(file,skiprows=1,delimiter=',',usecols=(0,1,-1))  
    #     selfie=np.repeat(chemstructure[molecule], data.shape[0]).reshape(52,data.shape[0]).T
    #     data=np.hstack((selfie,data))
    #     data_RASPA.append(data)
    
    for path_amount in path_IAST:
        path_amount = path_amount.replace("\\", "/")
        # print(path_amount)
        amount_mols = int(path_amount.split('/')[-1].split('_')[0])
        print(amount_mols)
        
        if amount_mols!=wanted_amount_mols:
            continue
        print("test")
        path_temps = glob.glob(path_amount + "/*")
        for folder in path_temps:
            folder = folder.replace("\\", "/") #comment this line if you use linux
            # print(folder)
            # print(folder.split('/')[-1].split("K")[0])
            temp = float(folder.split('/')[-1].split("K")[0])
            # print(folder + "/*.txt")
            molc_folder = glob.glob(folder + "/*")
            
            # print(molc_folder)
            for mols_mix in molc_folder:
                mols_mix = mols_mix.replace("\\", "/")
                # print(mols_mix)
                # print(type(amount_mols))
                mols = mols_mix.split('/')[-1].split('-')
                # print(molc_folder)
                path_fracs = glob.glob(mols_mix + "/*.txt")
                for file in path_fracs:
                    file = file.replace("\\", "/")
                    fracs = file.split('/')[-1].split('.txt')[0].split("-")
                    # print(fracs)
                    fracs = np.array(fracs, dtype=float)
                    # print(path_fracs)
                    # print(fracs)
                    try:
                        data=np.loadtxt(file,delimiter='   ',skiprows =1, usecols=(range(1,amount_mols+2)))#Pressure, loading m1, loading m2,..., loading mi
                    except:
                        print(f"failed to load file: {file}")
                        print(mols_mix)
                        break
                    # if len(mols)<wanted_amount_mols:
                    #     # print(data.shape)
                    #     # print(np.zeros((max_amount_mols - len(mols), len(data))).shape)
                    #     data = np.append(data, np.zeros((len(data),max_amount_mols - len(mols))), axis =1)
                    
                    
                    # print(data.shape)
                    selfie = 0
                    for i, mol in enumerate(mols):
                        selfie += np.repeat(chemstructure[mol], data.shape[0]).reshape(52,data.shape[0]).T
                    # print(selfie.shape)
                    # print(data.shape)
                    # temp_arr = np.full((len(data), 1), temp) 
                    # try:
                    #     selfie1 = np.repeat(chemstructure[mols[0]], data.shape[0]).reshape(52,data.shape[0]).T
                    #     selfie2 = np.repeat(chemstructure[mols[1]], data.shape[0]).reshape(52,data.shape[0]).T
                    #     data=np.hstack((np.full((len(data), 1), fracs[0]), selfie1,np.full((len(data), 1), fracs[1]), selfie2,temp_arr, data))
                    # except:

                    temp_arr = np.full((len(data), 1), temp) 
                    frac_arr = np.full((len(data), wanted_amount_mols), fracs)
                    data=np.hstack((selfie,frac_arr,temp_arr, data))
                    # data=np.hstack((selfie,temp_arr, data))
                    # selfie3 = np.repeat(chemstructure[mols[2]], data.shape[0]).reshape(52,data.shape[0]).T                    
                    data_IAST.append(data)
    data_IAST = np.vstack(data_IAST)
    x_vals=data_IAST[:,:-wanted_amount_mols] 
    y_vals=data_IAST[:,(len(data_IAST[0]))-wanted_amount_mols:]    
    return x_vals, y_vals

def make_training_database_ver4(wanted_amount_mols, chemstructure=ML_database()):
    path_IAST=glob.glob('IAST-segregated/automated_output/*')
    data_IAST=[]
    
    for path_amount in path_IAST:
        path_amount = path_amount.replace("\\", "/")
        amount_mols = int(path_amount.split('/')[-1].split('_')[0])
        
        if amount_mols!=wanted_amount_mols:
            continue

        path_temps = glob.glob(path_amount + "/*")
        for folder in path_temps:
            folder = folder.replace("\\", "/") #comment this line if you use linux
            
            temp = float(folder.split('/')[-1].split("K")[0])
            
            molc_folder = glob.glob(folder + "/*")
            for mols_mix in molc_folder:
                mols_mix = mols_mix.replace("\\", "/")

                mols = mols_mix.split('/')[-1].split('-')
                path_fracs = glob.glob(mols_mix + "/*.txt")
                for file in path_fracs:
                    file = file.replace("\\", "/")
                    fracs = file.split('/')[-1].split('.txt')[0].split("-")
                    fracs = np.array(fracs, dtype=float)
                    try:
                        data=np.loadtxt(file,delimiter='   ',skiprows =1, usecols=(range(1,amount_mols+2)))#Pressure, loading m1, loading m2,..., loading mi
                    except:
                        print(f"failed to load file: {file}")
                        continue

                    temp_arr = np.full((len(data), 1), temp) 
 
                    for it in range(amount_mols):
                        frac =  np.full((len(data), 1), fracs[it])
                        selfie = np.repeat(chemstructure[mols[it]], data.shape[0]).reshape(52,data.shape[0]).T
                        if it == 0:
                            new_data = np.hstack((frac, selfie))
                        else:
                            new_data = np.hstack((new_data, frac, selfie))

                    new_data = np.hstack((new_data,temp_arr,data))
                    data_IAST.append(new_data)
    return np.vstack(data_IAST)

def make_IAST_database_ver2(nummol, chemstructure=ML_database):
    """
    Enter the number of molecules you want to mix (nummol). The function will
    in default use the ML_database for the chemical structure of the molecules, 
    which is a dictionary containing all selfie structures of the used 
    molecules.
    The function will return the x and y values of the IAST-data:
     * The variable x_vals is an 2D array which contains of multiple arrays 
       filled with the following: fraction of mixture molecule
       1, selfies code molecule 1, ..., fraction of mixture molecule nummol, 
       selfies code molecule nummol, the pressure, the temprature. In each 
       array (inside x_vals) is either the pressure or temprature is varied.
     * The variable y_vals contains the following: loading of molecule 1, ...,
       loading of molecule nummol. 
       
    With the return values of this function you can create a train and test
    data set for your machine learning algorithm, using the following code 
    snippet:
        
    x_train, x_test, y_train, y_test= train_test_split(x_vals, y_vals,
                                            test_size= 0.1, random_state=0)  
    """
    #create list of all paths to the data files of the mixture of nummol molecules:
    path_IAST=glob.glob(f'IAST-segregated/automated_output/{nummol}_molecules/**/**/*.txt')
    
    chemstructure=ML_database()
    data_IAST =[]
    for file in path_IAST: 
        file = file.replace("\\", "/") #comment this line if you use linux
        folders=file.split('/') #creates a list of all folders in the directory leading to the file
        temp=int( folders[3][:3] ) #extracts the temperature of the mixture from the folders list
        
        #load data and insert the temperature at the end of each row of data:
        data=np.genfromtxt(file,delimiter='    ',skip_header=1,dtype=float) 
        data=np.insert(data,obj=1,axis=1,values=temp)
        
        for molnum in range(nummol):
            #create an array of the fraction (frac) and selfie (selfie) of 
            #molecule molnum and add it to the array frac_selfie, where the two
            #are combined. After the for loop the array frac_selfie contains
            #of rows filled with the fraction of molecule 1, selfie of molecule
            #1, ..., fraction of molecule nummol, selfie of nummol. frac_selfie
            #contains of the same amount of columns as the data array and each
            #row is identical.
            frac =  np.full((len(data), 1), float(folders[-1].split('-')[molnum].split(".txt")[0]))
            selfie = np.repeat(chemstructure[folders[4].split("-")[molnum]], data.shape[0]).reshape(52,data.shape[0]).T
            if molnum == 0:
                frac_selfie = np.hstack((frac, selfie))
            else:
                frac_selfie = np.hstack((frac_selfie, frac, selfie))
        
        #combine the frac_selfie array and the data array into one array. And 
        #append the combined array to data_IAST, which will gather all combined 
        #arrays:
        data=np.hstack((frac_selfie,data)) 
        data_IAST.append(data)
    
    data_IAST = np.vstack(data_IAST) #reorder the array
    
    #seperate the input values (x_vals) and output values (y_vals) from the 
    #combined data set (data_IAST):
    x_vals=data_IAST[:,:-nummol] 
    y_vals=data_IAST[:,(len(data_IAST[0]))-nummol:]    
    return x_vals, y_vals

# def make_IAST_database_ver3(min_nummol, max_nummol, chemstructure=ML_database):
#     """
#     Enter the number of molecules you want to mix (nummol). The function will
#     in default use the ML_database for the chemical structure of the molecules, 
#     which is a dictionary containing all selfie structures of the used 
#     molecules.
#     The function will return the x and y values of the IAST-data:
#      * The variable x_vals is an 2D array which contains of multiple arrays 
#        filled with the following: fraction of mixture molecule
#        1, selfies code molecule 1, ..., fraction of mixture molecule nummol, 
#        selfies code molecule nummol, the pressure, the temprature. In each 
#        array (inside x_vals) is either the pressure or temprature is varied.
#      * The variable y_vals contains the following: loading of molecule 1, ...,
#        loading of molecule nummol. 
       
#     With the return values of this function you can create a train and test
#     data set for your machine learning algorithm, using the following code 
#     snippet:
        
#     x_train, x_test, y_train, y_test= train_test_split(x_vals, y_vals,
#                                             test_size= 0.1, random_state=0)  
#     """
#     for nummol in range(min_nummol, max_nummol+1):
#         #create list of all paths to the data files of the mixture of nummol molecules:
#         path_IAST=glob.glob(f'IAST-segregated/automated_output/{nummol}_molecules/**/**/*.txt')
        
#         chemstructure=ML_database()
#         data_IAST =[]
#         for file in path_IAST: 
#             file = file.replace("\\", "/") #comment this line if you use linux
#             folders=file.split('/') #creates a list of all folders in the directory leading to the file
#             temp=int( folders[3][:3] ) #extracts the temperature of the mixture from the folders list
            
#             #load data and insert the temperature at the end of each row of data:
#             data=np.genfromtxt(file,delimiter='    ',skip_header=1,dtype=float) 
#             if nummol < max_nummol:
#                 data = np.insert(data,obj=1,axis=1,values=np.zeros(max_nummol - nummol))
            
#             data=np.insert(data,obj=1,axis=1,values=temp)
            
#             for molnum in range(nummol):
#                 #create an array of the fraction (frac) and selfie (selfie) of 
#                 #molecule molnum and add it to the array frac_selfie, where the two
#                 #are combined. After the for loop the array frac_selfie contains
#                 #of rows filled with the fraction of molecule 1, selfie of molecule
#                 #1, ..., fraction of molecule nummol, selfie of nummol. frac_selfie
#                 #contains of the same amount of columns as the data array and each
#                 #row is identical.
#                 frac =  np.full((len(data), 1), float(folders[-1].split('-')[molnum].split(".txt")[0]))
#                 selfie = np.repeat(chemstructure[folders[4].split("-")[molnum]], data.shape[0]).reshape(52,data.shape[0]).T
#                 if molnum == 0:
#                     frac_selfie = np.hstack((frac, selfie))
#                 else:
#                     frac_selfie = np.hstack((frac_selfie, frac, selfie))
                
#             if nummol < max_nummol:
#                 frac_selfie = np.hstack((frac_selfie, np.zeros((len(data), 53))))
                
#             #combine the frac_selfie array and the data array into one array. And 
#             #append the combined array to data_IAST, which will gather all combined 
#             #arrays:
#             data=np.hstack((frac_selfie,data)) 
#             data_IAST.append(data)
#         print(nummol)
#     data_IAST = np.vstack(data_IAST) #reorder the array
    
#     #seperate the input values (x_vals) and output values (y_vals) from the 
#     #combined data set (data_IAST):
#     x_vals=data_IAST[:,:-nummol] 
#     y_vals=data_IAST[:,(len(data_IAST[0]))-nummol:]    
#     return x_vals, y_vals

def make_IAST_database_ver3(nummol, chemstructure=ML_database):
    """
    Enter the number of molecules you want to mix (nummol). The function will
    in default use the ML_database for the chemical structure of the molecules, 
    which is a dictionary containing all selfie structures of the used 
    molecules.
    The function will return the x and y values of the IAST-data:
     * The variable x_vals is an 2D array which contains of multiple arrays 
       filled with the following: fraction of mixture molecule
       1, selfies code molecule 1, ..., fraction of mixture molecule nummol, 
       selfies code molecule nummol, the pressure, the temprature. In each 
       array (inside x_vals) is either the pressure or temprature is varied.
     * The variable y_vals contains the following: loading of molecule 1, ...,
       loading of molecule nummol. 
       
    With the return values of this function you can create a train and test
    data set for your machine learning algorithm, using the following code 
    snippet:
        
    x_train, x_test, y_train, y_test= train_test_split(x_vals, y_vals,
                                            test_size= 0.1, random_state=0)  
    """
    #create list of all paths to the data files of the mixture of nummol molecules:
    path_IAST=glob.glob(f'IAST-segregated/automated_output/{nummol}_molecules/**/**/*.txt')
    # chemstructure=ML_database
    data_IAST =[]
    for file in path_IAST: 
        file = file.replace("\\", "/") #comment this line if you use linux
        folders=file.split('/') #creates a list of all folders in the directory leading to the file
        temp=int( folders[3][:3] ) #extracts the temperature of the mixture from the folders list
        
        #load data and insert the temperature at the end of each row of data:
        data=np.genfromtxt(file,delimiter='    ',skip_header=1,dtype=float) 
        data=np.insert(data,obj=1,axis=1,values=temp)
        selfie = 0
        for molnum in range(nummol):
            #create an array of the fraction (frac) and selfie (selfie) of 
            #molecule molnum and add it to the array frac_selfie, where the two
            #are combined. After the for loop the array frac_selfie contains
            #of rows filled with the fraction of molecule 1, selfie of molecule
            #1, ..., fraction of molecule nummol, selfie of nummol. frac_selfie
            #contains of the same amount of columns as the data array and each
            #row is identical.
            frac =  np.full((len(data), 1), float(folders[-1].split('-')[molnum].split(".txt")[0]))
            # print(frac)
            selfie += np.repeat(chemstructure()[folders[4].split("-")[molnum]], data.shape[0]).reshape(52,data.shape[0]).T
            if molnum == 0:
                fracs = frac
            else:
                fracs = np.append(fracs, frac, axis=1)
            
        #combine the frac_selfie array and the data array into one array. And 
        #append the combined array to data_IAST, which will gather all combined 
        #arrays:
        # fracs_arr = np.full((len(data), nummol), fracs)
        data=np.hstack((selfie,fracs,data)) 
        data_IAST.append(data)
    
    data_IAST = np.vstack(data_IAST) #reorder the array
    
    #seperate the input values (x_vals) and output values (y_vals) from the 
    #combined data set (data_IAST):
    x_vals=data_IAST[:,:-nummol] 
    y_vals=data_IAST[:,(len(data_IAST[0]))-nummol:]    
    return x_vals, y_vals

def Performance(amount_mols, rf_model, x_train, x_test, y_train, y_test):
    y_pred=rf_model.predict(x_test) 
    
    
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
    
    plt.figure()
    plt.title(f"Relative error Decision Tree, {amount_mols} molecules micture")
    plt.scatter(range(len(rel_err)), rel_err, label="Relative error point i")
    plt.hlines(mean_rel_err, xmin = 0, xmax = len(rel_err), color="red", label=f"Mean relative error = {round(mean_rel_err,5)}")
    plt.yscale("log")
    plt.xlabel("Index of array rel_err")
    plt.ylabel("Relative error of predicted point wrt to known point")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title(f"Absolute error Decision Tree, {amount_mols} molecules micture")
    plt.scatter(range(len(abs_err)), abs_err, label="Absolute error point i")
    plt.hlines(mean_abs_err, xmin = 0, xmax = len(abs_err), color="red", label=f"Mean absolute error = {round(mean_abs_err,5)}")
    plt.yscale("log")
    plt.xlabel("Index of array abs_err")
    plt.ylabel("Absolute error of predicted point wrt to known point")
    plt.legend()
    
    plt.figure()
    plt.title(f"Performance Decision Tree, {amount_mols} molecules micture")
    plt.scatter(y_test, y_pred)
    plt.xlabel("True loading (from IAST)")
    plt.ylabel("Predicted loading (from Desicion Tree)")
    plt.show()
    
    print(f"Mean relative error = {mean_rel_err}")
    print(f"Mean relative error = {np.std(mean_rel_err)}")
    print(f"Mean absolute error = {mean_abs_err}")
    print(f"Mean absolute error = {np.std(mean_abs_err)}")


x_vals, y_vals = make_IAST_database_ver3(2, ML_database)
print("test")
x_train, x_test, y_train, y_test= train_test_split(x_vals, y_vals,
                                        test_size=0.1, random_state=0)  


regr = RandomForestRegressor(random_state=0, n_jobs=-1)
regr.fit(x_train, y_train)

# y_pred=regr.predict(x_test) 
Performance(3, regr, x_train, x_test, y_train, y_test)

# rel_err=np.abs(y_pred-y_test)#/y_test

# rel_err = rel_err[:,0]
# plt.figure()
# plt.title("Performance Decision Tree (VERSION 2)")
# plt.scatter(range(len(rel_err)), rel_err)
# plt.xlabel("Index of array rel_err")
# plt.ylabel("Relative error of predicted point wrt to known point")
# plt.show()

# plt.figure()
# plt.title("Performance Decision Tree (VERSION 2), zoomed in plot")
# plt.scatter(range(len(rel_err)), rel_err)
# plt.ylim(0,0.25)
# plt.xlabel("Index of array rel_err")
# plt.ylabel("Relative error of predicted point wrt to known point")
# plt.show()

# rel_err = rel_err[~np.isnan(rel_err)] #to remove nan's
# rel_err = rel_err[~np.isinf(rel_err)] #to remove inf's
# mean_rel_err = np.mean(rel_err)
# print(mean_rel_err)

# plt.figure()
# plt.title("Performance Decision Tree (VERSION 2), logaritmic plot")
# plt.scatter(range(len(rel_err)), rel_err, label="Relative error point i")
# plt.hlines(mean_rel_err, xmin = 0, xmax = len(rel_err), color="red", label="Mean relative error")
# plt.yscale("log")
# plt.xlabel("Index of array rel_err")
# plt.ylabel("Relative error of predicted point wrt to known point")
# plt.legend()
# plt.show()

# plt.figure()
# plt.title("Performance Decision Tree (VERSION 3),\nusing mix of three molecules")
# plt.scatter(y_test, y_pred)
# # plt.xscale("log")
# # plt.yscale("log")
# plt.xlabel("True loading (from IAST)")
# plt.ylabel("Predicted loading (from Desicion Tree)")
# plt.show()

# rel_err = rel_err[~np.isnan(rel_err)] #to remove nan's
# rel_err = rel_err[~np.isinf(rel_err)] #to remove inf's
# mean_rel_err = np.mean(rel_err)
# print(mean_rel_err)

# """Version 1, don't change is backup"""
# # data_set_raspa, data_set_iast = make_training_database()
    
# # x_vals=data_set_iast[:,:(len(data_set_iast[0]))-2]
# # y_vals=data_set_iast[:,(len(data_set_iast[0]))-2:]


# # x_train, x_test, y_train, y_test= train_test_split(x_vals, y_vals ,test_size= 0.1, random_state=0)  

# # regr = RandomForestRegressor(random_state=0)
# # regr.fit(x_train, y_train)

# # y_pred=regr.predict(x_test) 

# # rel_err=np.abs(y_pred-y_test)/y_test
# # rel_err = rel_err[:,0]
# # plt.figure()
# # plt.title("Performance Decision Tree (VERSION 1)")
# # plt.scatter(range(len(rel_err)), rel_err)
# # plt.xlabel("Index of array rel_err")
# # plt.ylabel("Relative error of predicted point wrt to known point")
# # plt.show()

# # plt.figure()
# # plt.title("Performance Decision Tree (VERSION 1), zoomed in plot")
# # plt.scatter(range(len(rel_err)), rel_err)
# # plt.ylim(0,0.25)
# # plt.xlabel("Index of array rel_err")
# # plt.ylabel("Relative error of predicted point wrt to known point")
# # plt.show()

# # rel_err = rel_err[~np.isnan(rel_err)] #to remove nan's
# # rel_err = rel_err[~np.isinf(rel_err)] #to remove inf's
# # mean_rel_err = np.mean(rel_err)
# # print(mean_rel_err)

# # plt.figure()
# # plt.title("Performance Decision Tree (VERSION 1), logaritmic plot")
# # plt.scatter(range(len(rel_err)), rel_err, label="Relative error point i")
# # plt.hlines(mean_rel_err, xmin = 0, xmax = len(rel_err), color="red", label="Mean relative error")
# # plt.yscale("log")
# # plt.xlabel("Index of array rel_err")
# # plt.ylabel("Relative error of predicted point wrt to known point")
# # plt.legend()
# # plt.show()




