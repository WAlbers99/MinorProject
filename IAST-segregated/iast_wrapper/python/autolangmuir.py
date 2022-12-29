#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Use this file to fit isotherms to the DS-Langmuir equation."
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
import os 
import subprocess
import pandas as df


plotpath = "../../langmuir_pure_plots/"
def load_raspa(temp, path = "../../../Raspa/outputs"):
    """Loads molecule names and paths from file dir
    Note: dependent on following filename:
    [molecule name]-[temperature]_out.txt NB "-" not optional."""
    mol_names = []
    mol_csvs = []
    for root, dir, files in os.walk(path):
        for names in files:
            if str(temp) in names:
                mol_names.append(names.partition("-")[0])
                #print(names)
                mol_csvs.append(root + "/" + names)         
    return mol_names, mol_csvs

def load_raspa_new(temp, mol_names, mol_csvs, path = "../../../Raspa/ShrinjayOutputs/ConvertedShrinjay"):
    """Loads molecule names and paths from file dir"""
    for root, dir, files in os.walk(path):
        for names in files:
            if str(temp) in names:
                mol_names.append(names.partition("-")[2])
                mol_csvs.append(root + "/" + names)     
    print(mol_csvs)      
    return mol_csvs, mol_names


def return_molkg_pressure(df_iso):
    """does exactly what it says on the tin"""
    try:
        molkg, pressure = df_iso["molkg"], df_iso["pressure"]
    except:
        molkg, pressure = df_iso["molkg"], df_iso["# pressure"]
    return molkg, pressure

def DSLangmuir(ab, k1, qsat1, k2, qsat2):
    k1ab, k2ab = k1 * ab, k2 * ab
    return (qsat1 * k1ab / ( 1 + k1ab)) + (qsat2 * k2ab / (1 + k2ab))

def try_curvefit(DSLangmuir, pressure, molkg, p0, maxfev = 2000):
    try: 
        popt, pcov = sp.optimize.curve_fit(DSLangmuir, pressure, molkg, p0=p0, maxfev = maxfev)
        p0 = popt
    except Exception as e: 
        print(e) 
        p0 = np.array([np.nan, np.nan, np.nan, np.nan])
    return p0

def iterative_DS_Langmuir(df_iso, k1_its=12, q_its=12, q1_value=0.7, maxfev = 3000): 
    "Generates various p0 starting values, then generates curvefits for all of them"
    qlinspace = np.linspace(0.3, 20, q_its)
    klogspace = np.logspace(1, -12, k1_its)
    molkg, pressure = return_molkg_pressure(df_iso)
    p0_array = np.zeros([4, k1_its * q_its * q_its])
    m = 0 #helper variable
    print("Started for-loop.")
    
    for i in range(0, k1_its):#loop first over k1,q1 
        for k in range(0, k1_its):
            for l in range (0, q_its):
                p0 = np.array([klogspace[i], q1_value, klogspace[k], qlinspace[l]])
                p0_array[:, m] = try_curvefit(DSLangmuir, pressure, molkg, p0=p0, maxfev = maxfev)
                m = m + 1
    return(p0_array)

def autoselect_p0_DS_Langmuir(data, name):
    "Selects data based upon two criteria: 1. no values should be smaller than 0;"
    "2. q1 should be between 0.6 and 0.8."
    "Returns the average p0 value."
    data = data[:, data.min(axis=0)>=10e-25] 
    if data[:, data[1, :] > data[3, :]].size == 0:
        data[[1, 3]] = data[[3, 1]]
        data[[0, 2]] = data[[2, 0]]
    data = data[:, data[1, :] > data[3, :]]
    data = np.delete(data, np.isnan(data).any(axis=0), axis=1)
    return np.average(data, axis=1)

def save_p0_plot(mol1iso, temp, sel_data, input1):
    plt.figure()
    mol1para = return_molkg_pressure(mol1iso)
    plt.loglog(mol1para[1], DSLangmuir(mol1para[1], sel_data[0], sel_data[1], sel_data[2], sel_data[3]), "r.", label=input1 + ", DS-Langmuir fit")
    plt.loglog(mol1para[1], mol1para[0], "go", label = input1 + ", RASPA-obtained data")
    plt.legend()
    plt.title("Dual-Side Langmuir fit of %s" %input1)
    plt.xlabel("Pressure (Pa)")
    plt.ylabel("Loading (mol/kg)")
    plt.savefig(plotpath + "%s-%d_plot" %(input1, temp))
    plt.close()

def check_if_iso_exists(temp, mol_paths, mol_names, path = "new_p0/"):
    try:
        with open(path + "p0_values-%d.txt" %(temp), "r") as file:
            for line in file:
                for s in mol_names:
                    if s in line:
                        del mol_paths[mol_names.index(s)]
                        mol_names.remove(s)
        return mol_paths, mol_names
    except:
        return mol_paths, mol_names

def auto_fit_plot_Langmuir(temp, calc_all = False, input_name = None, input_path = None):
    if input_name == None:
        if input_path == None:
            paths, mol_names = load_raspa(temp)
            paths, mol_names = load_raspa_new(temp, paths, mol_names)
    else:
        paths, mol_names = [input_path], [input_name]
    output = []
    if calc_all == False:
        paths, mol_names = check_if_iso_exists(temp, paths, mol_names)
    for i in range(0, len(mol_names)):
        print(mol_names[i])
        #mol_1_iso = df.read_csv(paths[i])
        mol_1_iso, name = df.read_csv(paths[i]), mol_names[i]
        mol1para = return_molkg_pressure(mol_1_iso)
        sel_data = autoselect_p0_DS_Langmuir(iterative_DS_Langmuir(mol_1_iso), name)
        output.append(sel_data)
        save_p0_plot(mol_1_iso, temp, sel_data, name)
    output1 = np.zeros(4)
    if calc_all == False:
        if os.path.exists('new_p0/p0_values-%d.txt') == True:
            p0_append_write = "a"
        else:
            p0_append_write = "w"
    else:
        p0_append_write = "w"
    with open('new_p0/p0_values-%d.txt' %temp, p0_append_write) as f:
        for i in range(0, len(output)):
            output1 = np.array( output[i])
            f.write("%s \t %s \n" %(mol_names[i], output1))
    return 0;


def input_wrapper_langmuir():
    auto_fit_plot_Langmuir(input("Please input a temperature"))
#name = "22mC5"
#paths = "../../../Raspa/%s/%s-%dout.txt" %(name, name, temp)
def main():
    temp = 550
    auto_fit_plot_Langmuir(temp, calc_all=True)
    #input_wrapper_langmuir()
    #molkg, pressure = return_molkg_pressure(df.read_csv("/home/mike/Documents/uni/alice/Final_project/MinorProject/Raspa/outputs/3mC5/3mC5-600out.txt"))
if __name__ == "__main__":
    main()  