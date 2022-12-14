#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 10:52:33 2022

@author: mike
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyiast 


def return_langmuir(df_iso):
    return pyiast.ModelIsotherm(df_iso, loading_key = "molkg", pressure_key = "pressure", model = "Langmuir")
#verander onderstaand NIET omdat mijn computer te ghetto is voor een andere oplossing
#df_ch4 = pd.read_csv("/home/mike/Documents/uni/alice/Final_project/MinorProject/iast/input/ethaan_iso.txt")
#df_ch3ch3 = pd.read_csv("/home/mike/Documents/uni/alice/Final_project/MinorProject/iast/input/methane_iso2.txt")
input_path = "../Raspa/outputs/"
#df_ch4 = pd.read_csv("./input/ethaan_iso.txt")
#df_ch3ch3 = pd.read_csv("./input/methane_iso2.txt")

#df_c7_300 = pd.read_csv(input_path + "2mC6/2mC6-400")
df_c7_300 = pd.read_csv(input_path + "C7/C7-400out.txt")
df_c7_400 = pd.read_csv(input_path + "3mC6/3mC6-400out.txt")

c7_300_iso = return_langmuir(df_c7_300)
c7_400_iso = return_langmuir(df_c7_400)

#Define labels 

molecule_1 = r"$CH_7$"
molecule_2 = r"$3mC6$"
temp = r"$500 K$"

# Fit Langmuir isotherm to dataframe 

#ch4_isotherm = pyiast.ModelIsotherm(df_ch4, loading_key = "molkg", pressure_key = "pressure", model = "Langmuir")
#ch3ch3_isotherm = pyiast.ModelIsotherm(df_ch3ch3, loading_key = "molkg", pressure_key = "pressure", model = "Langmuir")


#pyiast.plot_isotherm(ch3ch3_isotherm)
#pyiast.plot_isotherm(ch4_isotherm)

#ch4_isotherm.print_params()
#ch3ch3_isotherm.print_params()

#Export ch4 ch3 isotherm to .txt 



#gas_frac = np.array([0.10, 0.90])
#total_pressure = np.linspace(0, 10e4, 50)
#mixture = np.zeros([50, 2])
#for i in range(0,50):
#    mixture[i] = pyiast.iast(total_pressure[i] * gas_frac, [ch4_isotherm, ch3ch3_isotherm])


#plt.semilogx(total_pressure, mixture[:, 0], "r+", label=r"$CH_4$, IAST-approx. mixture")
#plt.semilogx(total_pressure, mixture[:, 1], "g+", label=r"$CH_3-CH_3$, IAST-approx. mixture")
#plt.semilogx(total_pressure, ch4_isotherm.loading(total_pressure), "ro", label=r"$CH_4$, homogeneous gas")
#plt.semilogx(total_pressure, ch3ch3_isotherm.loading(total_pressure), "go", label=r"$CH_3-CH_3$, homogeneous gas")
#plt.title("Loading of a %1.0f - %1.0f percent methane/ethane mixture" %(gas_frac[0]*100, gas_frac[1]*100))
#plt.xlabel("Pressure (bar)")
#plt.ylabel("Loading (mol/kg)")
#plt.legend()
#plt.show()

# Now doing the same but for multiple mixtures: 
no_fracs = 2
no_pressures = 20
gas_frac = np.linspace(0.5, 0.5, no_fracs)
partial_pressures = np.logspace(0, 9, num=no_pressures)
mix_isotherm = np.zeros((no_fracs, no_pressures, 2))
for i in range(0, no_fracs):
    for j in range(0, no_pressures):
        mix_isotherm[i, j] = pyiast.iast(partial_pressures[j] * np.array([gas_frac[i], 1-  gas_frac[i]]), [c7_300_iso, c7_400_iso])

print(partial_pressures)
plt.figure()
for i in range(1, no_fracs-1):
    pass
    #plt.semilogx(partial_pressures, mix_isotherm[i,:, 0], "r+")
    #plt.semilogx(partial_pressures, mix_isotherm[i,:, 1], "g+")
#plt.semilogx(partial_pressures, mix_isotherm[1,:, 0], "rv", label=molecule_1 +r", lowest fraction")
#plt.semilogx(partial_pressures, mix_isotherm[1,:, 1], "gv", label=molecule_2 + r", highest fraction")
#plt.semilogx(partial_pressures, mix_isotherm[no_fracs-1,:, 0], "r^", label= molecule_1 + r", highest fraction")
#plt.semilogx(partial_pressures, mix_isotherm[no_fracs-1,:, 1], "g^", label= molecule_2 + r", lowest fraction")
plt.semilogx(partial_pressures, c7_300_iso.loading(partial_pressures), "ro", label=molecule_1 + r", homogeneous gas")
plt.semilogx(partial_pressures, c7_400_iso.loading(partial_pressures), "go", label=molecule_2 + r", homogeneous gas")
plt.title("Loadings of a " + molecule_1 + " and " + molecule_2 +  " mixture at temperature " + temp)
plt.xlabel("Pressure (bar)")
plt.ylabel("Loading (mol/kg)")
#plt.legend()
plt.show()

c7_300_iso.print_params()
c7_400_iso.print_params()

#Note: at low pressure longer molecules have higher loading; at high pressure shorter molecues have higher loading. 

#Write results to an output file

np.savetxt("output/pressures_gas_frac.txt", partial_pressures)
np.savetxt("output/mixing_fractures.txt", gas_frac)
np.savetxt("output/mix_isothermc7-300.csv", mix_isotherm[:,:,0])
np.savetxt("output/mix_isothermc7-500.csv", mix_isotherm[:,:,1])