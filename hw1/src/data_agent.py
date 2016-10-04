import numpy as np
from datatype import obv_data

def input_training_data(adr):
    itdata = np.genfromtxt(adr, delimiter = ",", usecols = range(3,27), dtype = 'str', skip_header = 1)
    dataset = []
    for i in range(len(itdata)/18/20):
        idata = itdata[(i * 360):((i + 1) * 360)]
        ten_hours_counter = 0
        data = obv_data()
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
                dataset.append(data)
                data = obv_data()
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
                data.add_oh_para([0.0 if i == 10 and idata[row_index + i][col] == "NR" else float(idata[row_index + i][col]) for i in range(18)], True)
            else:
                data.add_oh_para([0.0 if i == 10 and idata[row_index + i][col] == "NR" else float(idata[row_index + i][col]) for i in range(18)], True if ten_hours_counter == 9 else False)
            ten_hours_counter += 1
            col += 1
        dataset.append(data)
    return dataset

def input_testing_data(adr):
    idata = np.genfromtxt(adr, delimiter = ",", usecols = [0] + range(2,11), dtype = "str")
    dataset = []
    for row_index in range(0, len(idata), 18):
        data = obv_data(idata[row_index][0])
        for col in range(1,10):
            data.add_oh_para([0.0 if i == 10 and idata[row_index + i][col] == "NR" else float(idata[row_index + i][col]) for i in range(18)], False)
        dataset.append(data)
    return dataset

def output_result(adr, data):
    odata = open(adr, 'w')
    odata.write("id,value\n")
    for i in data:
        odata.write("{0},{1}\n".format(i.name, int(i.function_ans)))


