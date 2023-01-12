import numpy as np
import matplotlib.pyplot as plt

amount_mols = [2,3,4]

MRE_DT = [0.061963025491574855, 0.09232784290827868, 0.09704720524759813]
test_time_DT = [3.36e-16, 4.33e-09, 7.94e-09] #test for 1 molmix

MRE_RF = [0.058139046049532274, 0.08339960814726001, 0.0840687848090876]
test_time_RF = [1.56e-07, 1.74e-07, 4.11e-04]

MRE_kNN = [0.17413252987435657,0.17086368027816196,0.11083863534636515]
test_time_kNN = [6.78e-04,3.53e-03,8.19e-02]

MRE_SVR = 147.75694273036007
test_time_SVR = 1.08e-04

test_time_IAST = [22e-3,21e-3,22e-3]

MRE_NN = 0
test_time_NN = 0

plt.figure()
plt.title("Relative error of Machine Learning models")
plt.xlabel("Amount of molecules in mixture")
plt.ylabel("Relative error")
plt.scatter(amount_mols, MRE_DT, marker = "o", label="Decision Tree")
plt.scatter(amount_mols, MRE_RF, marker = "o", label="Random Forest")
plt.scatter(amount_mols, MRE_kNN, marker = "o", label="k-Nearest Neighbours")
plt.scatter(2, MRE_SVR, marker = "o", label="Support Vector Regression")
# plt.scatter(2, MRE_NN, marker = "o", label="Neural Network")
plt.yscale("log")
plt.legend()
plt.savefig("RelErr_ALlModels")
plt.show()

plt.figure()
plt.title("Relative error of Machine Learning models, zoomed in")
plt.xlabel("Amount of molecules in mixture")
plt.ylabel("Relative error")
plt.scatter(amount_mols, MRE_DT, marker = "o", label="Decision Tree")
plt.scatter(amount_mols, MRE_RF, marker = "o", label="Random Forest")
plt.scatter(amount_mols, MRE_kNN, marker = "o", label="k-Nearest Neighbours")
plt.yscale("log")
plt.legend()
plt.savefig("RelErr_ALlModels_zoomedIn")
plt.show()

plt.figure()
plt.title("Prediction time of one mixture of molecule\nfor the Machine Learning models and IAST")
plt.xlabel("Amount of molecules in mixture")
plt.ylabel("Prediction time 1 mixture (s)")
plt.scatter(amount_mols, test_time_DT, label="Decision Tree")
plt.scatter(amount_mols, test_time_RF, label="Random Forest")
plt.scatter(amount_mols, test_time_kNN, label="k-Nearest Neighbours")
plt.scatter(2, test_time_SVR, label="Support Vector Regression")
# plt.scatter(2, test_time_NN, label="Neural Network")
plt.scatter(amount_mols, test_time_IAST, label="IAST")
plt.yscale("log")
plt.legend()
plt.savefig("PredTime_AllModels")
plt.show()

