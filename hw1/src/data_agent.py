import sys
import numpy as np
from datatype import dataset

def input_training_data(adr, sh, hr):
    itdata = np.genfromtxt(adr, delimiter = ",", usecols = range(3,27), dtype = 'str', skip_header = 1)
    data = dataset(sh, hr)
    for i in range(len(itdata)/18/20):
        idata = itdata[(i * 360):((i + 1) * 360)]
        ten_hours_counter = 0
        pre_col = 0
        col = 0
        pre_ri = 0
        row_index = 0
        no_overlap = True
        while (1):
            if row_index == len(idata):
                break
            if col == 24:
                row_index += 18
                col = 0
                continue

            if ten_hours_counter == 10:
                data.append()
                ten_hours_counter = 0
                if pre_col == 23:
                    pre_col = 0
                    pre_ri += 18
                else:
                   pre_col += 1
                if no_overlap:
                    no_overlap = False
                col = pre_col
                row_index = pre_ri
                continue
            if no_overlap:
                data.add_single_hour([0.0 if i == 10 and idata[row_index + i][col] == "NR" else float(idata[row_index + i][col]) for i in range(18)], True, True if ten_hours_counter == 9 else False)
            else:
                data.add_single_hour([0.0 if i == 10 and idata[row_index + i][col] == "NR" else float(idata[row_index + i][col]) for i in range(18)], True if ten_hours_counter == 9 else False, True if ten_hours_counter == 9 else False)
            ten_hours_counter += 1
            col += 1
        data.append()
    return data

def input_testing_data(adr, sh, hr, val):
    idata = np.genfromtxt(adr, delimiter = ",", usecols = [0] + range(2,11 if val == 0 else 12), dtype = "str")
    data = dataset(sh, hr)
    for row_index in range(0, len(idata), 18):
        data.add_name(idata[row_index][0])
        for col in range(1,10 if val == 0 else 11):
            data.add_single_hour([0.0 if i == 10 and idata[row_index + i][col] == "NR" else float(idata[row_index + i][col]) for i in range(18)], False, False if col != 10 else True)
        data.append()
    return data

def output_result(adr, data):
    odata = open(adr, 'w')
    odata.write("id,value\n")
    name_set = data.get_name()
    function_ans_set = data.get_f_ans()
    for i in range(data.get_size()):
        odata.write("{0},{1}\n".format(name_set[i], int(function_ans_set[i, 0])))
    odata.close()
