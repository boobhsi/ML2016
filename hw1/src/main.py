from data_agent import *
from datatype import *
from sys import argv
import numpy as np
import csv

#define feature count
var_shr_count = 18
hr_count = 9

#input data
data = input_training_data(argv[1], var_shr_count, hr_count)

#order is a list recording order of every feature
order = np.array([int(argv[5])] * var_shr_count * hr_count)

#automatically build a array recording order
max_order = np.amax(order)
model_order = np.array([1 if order[i] > 0 else 0 for i in range(var_shr_count * hr_count)])[np.newaxis, :, np.newaxis]
order -= 1
for j in range(1, max_order):
    model_order = np.dstack((model_order, np.array([1 if order[i] > 0 else 0 for i in range(var_shr_count * hr_count)])[np.newaxis, :, np.newaxis]))
    order -= 1

#set all coefficients and bias to zero
coe = np.zeros((1, var_shr_count * hr_count, max_order))
bias = 0

#set learning rate
learning_rate_of_coe = float(argv[6])

#set regularization
smoother = float(argv[8])

#set iteration number
iteration = int(argv[2])

#normalize features
"""
dataset.get_global_sta()
data.normalize()
"""

#build leaner equation
equ = linear_equ(model_order, coe, bias, max_order, data.get_size())

#use linear equation to guess answer
data.refresh_ans(equ)

#transpose training answer
data.arraize()

#implement adagrad
accu_grad_squ = np.zeros((1, var_shr_count * hr_count, max_order))
accu_bias_squ = 0.0

#record error
err = 0.0

#edata = open("./err_record.csv", "a")

#main function
while iteration > 0:
    temp_string = ""

    #calculate gradient of coefficients
    err_coe = equ.err_pd_coe(data)

    #regularization if needed
    err_coe += 2 * smoother * coe

    #calculate gradient 0of bias
    err_bias = equ.err_pd_bias(data)

    #use adagrad or not
    if int(argv[7]) == 1:
        temp_string += "Using adagrad, "
        accu_grad_squ += err_coe ** 2
        accu_bias_squ += err_bias ** 2
        coe -= learning_rate_of_coe * err_coe / (accu_grad_squ ** 0.5)
        bias -= learning_rate_of_coe * err_bias / (accu_bias_squ ** 0.5)
    else:
        coe -= learning_rate_of_coe * err_coe
        bias -= learning_rate_of_coe * err_bias

    temp_string += ("remain " + str(iteration) + " times -> ")

    #change coefficient and bias
    equ.change_coe(coe, bias, model_order)

    #refresh y^
    data.refresh_ans(equ)

    #calculate error
    err = round(equ.err_by_data(data), 5)
    temp_string += ("err = " + str(err))
    print temp_string

    #to next iteration
    iteration -= 1

#for experiment
"""
    if iteration == int(argv[2]) - 1:
        edata.write(str(err) + ",")
    elif iteration % 50 == 0:
        if iteration == 0 and int(argv[9]) == 0:
            edata.write(str(err) + "\n")
        else:
            edata.write(str(err) + ",")
"""

#input testing data and calculate error of testing data
testing_data = input_testing_data(argv[3], var_shr_count, hr_count, 0)
testing_data.refresh_ans(equ)
testing_data.arraize()

#validation_mode for experiment
"""
if int(argv[9]) == 1:
    edata.write(str(err) + "," + str(round(equ.err_by_data(testing_data), 5)) + "\n")

edata.close()
"""

#output_result
output_result(argv[4], testing_data)
