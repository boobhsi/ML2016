from data_agent import *
from datatype import *
from sys import argv
import random

model_order = [1] * 8 * 18 + [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
learning_rate = 0.0000017
smoother = 0.05

coe_count = sum(model_order) + 1
random.seed()
coe = [0] * coe_count
for i in range(1,coe_count):
    coe[i] = random.uniform(-0.000001, 0.000001)
coe[0] = random.uniform(0,20)

equ = linear_equ(obv_data.var_count,
                 model_order,
                 coe)
iteration = int(argv[2])
dataset = input_training_data(argv[1])
#dataset[len(dataset)-1].normalize()
for i in dataset:
    i.refresh_ans(equ)

while iteration > 0:
    print iteration
    for coe_index in range(len(coe)):
        err = 0.0
        for data_index in range(len(dataset)):
            err += equ.err_pd_coe(coe_index, dataset[data_index])
        if coe_index != 0:
            err += 2 * smoother * coe[coe_index]
        #print err
        coe[coe_index] -= (learning_rate / len(dataset) * err)
    equ.change_coe(coe)
    #equ.print_coe()
    t_err = 0.0
    for data_index in range(len(dataset)):
        dataset[data_index].refresh_ans(equ)
        #print dataset[data_index].function_ans
        t_err += equ.err_by_data(dataset[data_index])
    print round(t_err/len(dataset), 2)
    iteration -= 1

testing_data = input_testing_data(argv[3])
#print testing_data[0]
#print testing_data[len(testing_data)-1]
t_err = 0.0
for i in testing_data:
    i.set_f_pm(equ)
    #t_err += equ.err_by_data(i)
#print round(t_err/len(testing_data), 2)
output_result(argv[4], testing_data)

