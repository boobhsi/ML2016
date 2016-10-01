import numpy as np
from datatype import obv_data

def gen_new_chain():
    data_chain = []
    for index in range(24):
        data_chain.append(obv_data())
    return data_chain


def input_data(adr):
    idata = np.genfromtxt(adr, delimiter = ",", usecols = range(3,27), dtype =
                          'str', skip_header = 1)
    count = 0
    dataset = []
    data_chain = []

    for row in idata:
        if count == 18:
            dataset.append(data_chain)
            count = 0
        for col in range(24):
            if count == 0 and col == 0:
                data_chain = gen_new_chain()
            if count == 10:
                data_chain[col].add_para(0.0 if row[col] == "NR" else
                                    float(row[col]))
            elif count == 9:
                data_chain[col].set_pm(float(row[col]))
            else:
                #print float(row[col])
                data_chain[col].add_para(float(row[col]))
        count += 1
    return dataset




