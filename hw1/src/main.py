from data_agent import *
from datatype import *
from datatype2 import *
from sys import argv
import random
import numpy as np

#model_order = np.repeat([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],], 9, 0)
var_shr_count = 18
hr_count = 9

data = input_training_data(argv[1], var_shr_count, hr_count)
print data
#print data.get_var().shape

order = np.array([int(argv[5])] * var_shr_count * hr_count)
max_order = np.amax(order)
model_order_2d = [None] * max_order
model_order = None
coe = None
model_order = np.array([1 if order[i] > 0 else 0 for i in range(var_shr_count * hr_count)])[np.newaxis, :, np.newaxis]
order -= 1
for j in range(1, max_order):
    model_order = np.dstack((model_order, np.array([1 if order[i] > 0 else 0 for i in range(var_shr_count * hr_count)])[np.newaxis, :, np.newaxis]))
    order -= 1
#print model_order_2d[0]
#coe_2d[0] = np.random.uniform(-0.01, 0.01, (9, 18))
coe = np.zeros((1, var_shr_count * hr_count, max_order))
#print coe_2d[0]
#coe_2d[1] = np.random.uniform(-0.001, 0.001, (9, 18)) * np.repeat([[1 for i in range(var_col_count)],], var_row_count, 0)

#print model_order.shape

#learning_rate_of_coe = 0.01
learning_rate_of_coe = float(argv[6])
#learning_rate_of_weight = 0.00005
#weight_count = 10
smoother = int(argv[8])
#coe = np.random.uniform(-0.01, 0.01, (9, 18))
#print coe.shape

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
dataset.get_global_sta()
data.normalize()
#bias = obv_data.average_ft[9]
bias = 0

equ = linear_equ(model_order, coe, bias, max_order, data.get_size())
data.refresh_ans(equ)
data.arraize()

accu_grad_squ = np.zeros((1, var_shr_count * hr_count, max_order))
accu_bias_squ = 0.0
err = 0.0

while iteration > 0:
    temp_string = ""
    #print "remaining " + str(iteration) + " times"
    """
    err_coe = np.zeros((var_row_count, var_col_count, max_order) if max_order != 1 else (var_row_count, var_col_count))
    err_bias = 0.0
    for data_index in range(len(dataset)):
        err_coe += equ.err_pd_coe(dataset[data_index])
        err_bias += equ.err_pd_bias(dataset[data_index])
        #if coe_index != 0:
            #err += 2 * smoother * coe[coe_index]
        #print err
    err_coe += 2 * smoother * coe
    #print err_coe
    #print err_bias
    """
    err_coe = equ.err_pd_coe(data)
    err_coe += 2 * smoother * coe
    err_bias = equ.err_pd_bias(data)
    if int(argv[7]) == 1:
        temp_string += "Using adagrad, "
        accu_grad_squ += err_coe ** 2
        accu_bias_squ += err_bias ** 2
        coe -= learning_rate_of_coe / data.get_size() * err_coe / (accu_grad_squ ** 0.5)
        bias -= learning_rate_of_coe / data.get_size() * err_bias / (accu_bias_squ ** 0.5)
    else:
        coe -= learning_rate_of_coe / data.get_size() * err_coe
        bias -= learning_rate_of_coe / data.get_size() * err_bias

    temp_string += ("remaining " + str(iteration) + " times -> ")
    #print "adagrad of bias: {0}".format(learning_rate_of_coe / accu_bias_squ ** 0.5)
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
    """
    t_err = 0.0
    for data_index in range(len(dataset)):
        dataset[data_index].refresh_ans(equ)
    """
        #print dataset[data_index].function_ans
        #print dataset[data_index].function_ans
    data.refresh_ans(equ)
    err = round(equ.err_by_data(data), 2)
    temp_string += ("err = " + str(err))
    print temp_string
    iteration -= 1

testing_data = input_testing_data(argv[3], var_shr_count, hr_count)
#print testing_data[0]
#print testing_data[len(testing_data)-1]
testing_data.normalize()
testing_data.refresh_ans(equ)
#t_err += equ.err_by_data(i)
#print round(t_err/len(testing_data), 2)
output_result(argv[4], testing_data, err)

