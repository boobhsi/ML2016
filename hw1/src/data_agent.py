import numpy as np
from datatype import obv_data

def input_training_data(adr):
    idata = np.genfromtxt(adr, delimiter = ",", usecols = range(3,27), dtype = 'str', skip_header = 1)
    ten_hours_counter = 0
    dataset = []
    data = obv_data()
    for row_index in range(0, len(idata), 18):
        for col in range(24):
            if ten_hours_counter == 10:
                dataset.append(data)
                data = obv_data()
                ten_hours_counter = 0
            data.add_oh_para([0.0 if i == 10 and idata[row_index + i][col] == "NR" else float(idata[row_index + i][col]) for i in range(18)])
            ten_hours_counter += 1
    dataset.append(data)
    return dataset

def input_testing_data(adr):
    idata = np.genfromtxt(adr, delimiter = ",", usecols = [0] + range(2,11), dtype = "str")
    dataset = []
    for row_index in range(0, len(idata), 18):
        data = obv_data(idata[row_index][0])
        for col in range(1,10):
            data.add_oh_para([0.0 if i == 10 and idata[row_index + i][col] == "NR" else float(idata[row_index + i][col]) for i in range(18)])
        dataset.append(data)
    return dataset

def output_result(adr, data):
    odata = open(adr, 'w')
    odata.write("id,value\n")
    for i in data:
        odata.write("{0},{1}\n".format(i.name, int(i.function_ans)))


