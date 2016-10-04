from data_agent import *
from datatype import *
from sys import argv
import random
import numpy as np

model_order = np.repeat([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],], 9, 0)
learning_rate_of_coe = 0.015
#learning_rate_of_weight = 0.00005
#weight_count = 10
smoother = 1
coe = np.random.uniform(-0.01, 0.01, (9, 18))
print coe.shape

#coe_count = sum(model_order) + 1
#random.seed()
#coe = [0.0] * coe_count
#weight = [0.0] * weight_count

#define coe

"""
for i in range(1,weight_count):
    weight[i] = random.uniform(-0.1,0.1)
weight[0] = random.uniform(-10,10)
"""

#equ = designed_equ_1(obv_data.para_count, model_order, coe, weight_count - 1, weight)

iteration = int(argv[2])
dataset = input_training_data(argv[1])
print dataset[len(dataset)-2]
dataset[len(dataset)-1].get_global_sta()
bias = obv_data.average_ft[9]
for i in dataset:
    i.normalize()

equ = linear_equ(model_order, coe, bias)

for i in dataset:
    i.refresh_ans(equ)

while iteration > 0:
    print str(iteration) + " times"
    err_coe = np.zeros((9, 18))
    err_bias = 0.0
    for data_index in range(len(dataset)):
        err_coe += (equ.err_pd_coe(dataset[data_index]) + 2 * smoother * coe )
        err_bias += equ.err_pd_bias(dataset[data_index])
        #if coe_index != 0:
            #err += 2 * smoother * coe[coe_index]
        #print err
    coe -= (learning_rate_of_coe / len(dataset) * err_coe)
    bias -= (learning_rate_of_coe / len(dataset) * err_bias)
    """
    for weight_index in range(len(weight)):
        err = 0.0
        for data_index in range(len(dataset)):
            err += equ.err_pd_weight(weight_index, dataset[data_index])
        weight[weight_index] -= (learning_rate_of_weight / len(dataset) * err)

    """
    equ.change_coe(coe, bias)
    #equ.change_weight(weight)
    #equ.print_coe()
    #equ.print_weight()
    t_err = 0.0
    for data_index in range(len(dataset)):
        dataset[data_index].refresh_ans(equ)
        #print dataset[data_index].function_ans
        #print dataset[data_index].function_ans
        t_err += equ.err_by_data(dataset[data_index])
    print round((t_err/len(dataset)) ** 0.5, 2)
    iteration -= 1

testing_data = input_testing_data(argv[3])
#print testing_data[0]
#print testing_data[len(testing_data)-1]
for i in testing_data:
    i.arraize_para()
    i.set_f_pm(equ)
    #t_err += equ.err_by_data(i)
#print round(t_err/len(testing_data), 2)
output_result(argv[4], testing_data)

