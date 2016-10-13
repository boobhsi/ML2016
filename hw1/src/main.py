from data_agent import *
from datatype import *
from sys import argv
import random
import numpy as np

var_shr_count = 18
hr_count = 9

data = input_training_data(argv[1], var_shr_count, hr_count)

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

coe = np.zeros((1, var_shr_count * hr_count, max_order))

learning_rate_of_coe = float(argv[6])
smoother = int(argv[8])
iteration = int(argv[2])

dataset.get_global_sta()
data.normalize()
bias = 0

equ = linear_equ(model_order, coe, bias, max_order, data.get_size())
data.refresh_ans(equ)
data.arraize()

accu_grad_squ = np.zeros((1, var_shr_count * hr_count, max_order))
accu_bias_squ = 0.0
err = 0.0

while iteration > 0:
    temp_string = ""
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
    equ.change_coe(coe, bias)
    data.refresh_ans(equ)
    err = round(equ.err_by_data(data), 2)
    temp_string += ("err = " + str(err))
    print temp_string
    iteration -= 1

testing_data = input_testing_data(argv[3], var_shr_count, hr_count)
testing_data.normalize()
testing_data.refresh_ans(equ)
output_result(argv[4], testing_data, err)

