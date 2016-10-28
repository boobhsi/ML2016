import numpy as np
from sys import argv
from neuron import *
import pickle

def normalize(data):
    data_s = data ** 2
    average = np.sum(data, axis = 0) / data.shape[0]
    sd = (np.sum(data_s, axis = 0) / data.shape[0] - average ** 2) ** 0.5
    average = np.repeat(average[np.newaxis, :], data.shape[0], axis = 0)
    sd = np.repeat(sd[np.newaxis, :], data.shape[0], axis = 0)
    return (data - average) / sd

if int(argv[1]) == 1:
    #readfile for testing
    model_data =open(argv[2], "rb")
    ne = pickle.load(model_data)
    model_data.close()
    #print argv[3]
    tdata = np.genfromtxt(argv[3], delimiter = ",", dtype = "float", usecols = range(1, 58))
    tdata = np.array(tdata)
    tdata = normalize(tdata)
    ans = ne.get_ans(tdata)
    odata = open(argv[4], "w")
    odata.write("id,label\n")
    for i in range(ans.shape[0]):
        odata.write("{0},{1}\n".format(i+1, 1 if ans[i][0] == 1 else 0))

else:
    #define feature count
    fc = 57

    #readfile
    data = np.genfromtxt(argv[2], delimiter = ",", dtype = "float", usecols = range(1, 58))
    train_ans = np.genfromtxt(argv[2], delimiter = ",", dtype = "int", usecols = range(58, 59))

    data = np.array(data)
    train_ans = np.transpose(np.array(train_ans, ndmin = 2))

    #print data
    #print train_ans

    data = normalize(data)
    #init
    ne = neuron(fc)

    #iteration
    iteration = int(argv[3])

    #lr
    learning_rate = float(argv[4])

    #init_ans
    ne.init_ans(data)

    while(iteration > 0):
        iteration -= 1
        ne.refresh_para(data, learning_rate, train_ans)
        #print "{0}th iteration: ce = {1}".format(int(argv[3]) - iteration, ne.cal_cross_entropy(train_ans))
	#print "{0}th iteration".format(int(argv[3]) - iteration)

    #output_neuron
    with open(argv[5], "wb") as output:
        pickle.dump(ne, output, pickle.HIGHEST_PROTOCOL)
