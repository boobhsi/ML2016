import numpy as np
import random

class neuron:
    def __init__(self, fc):
        self.fc = fc
        self.omega = np.random.rand(fc, 1)
        self.bias = 0.0
        self.new_ans = None

    def cal_z(self, data):
        return np.dot(data, self.omega) - self.bias

    def cal_temp_z(self, data, omega, bias):
        return np.dot(data, omega) - bias

    #def cal_pro(self, data):
    #    return 1 / (1 + np.exp((self.cal_z(data) * -1)))

    def refresh_para(self, data, ta, iteration):
        least_false = self.count_false(self.cal_temp_z(data, self.omega, self.bias), ta)
        total_counter = 0
        global_omega = self.omega
        global_bias = self.bias
        while iteration > 0:
            i = random.randint(0, data.shape[0] - 1)
            temp = self.cal_temp_z(data[i], global_omega, global_bias)[0]
            if (temp > 0 and ta[i][0] == 0) or (temp < 0 and ta[i][0] == 1):
                false_count = 0
                if temp > 0:
                    global_omega = global_omega - np.transpose(data[i])
                    global_bias += 1
                else:
                    global_omega = global_omega + np.transpose(data[i])
                    global_bias -= 1
                temp_ans = self.cal_temp_z(data, global_omega, global_bias)
                false_count = self.count_false(temp_ans, ta)
                if false_count <= least_false:
                    least_false = false_count
                    self.omega = global_omega
                    self.bias = global_bias
                total_counter += 1
                iteration -= 1
                #print "{0} : {1} vs {3} at {4}, bias = {2}".format(total_counter, least_false, self.bias, false_count, i)

    def count_false(self, ans, ta):
        false_count = 0
        for i in range(ans.shape[0]):
            if (ans[i][0] > 0 and ta[i][0] == 0) or (ans[i][0] < 0 and ta[i][0] == 1): false_count += 1
        return false_count


    #def init_ans(self, data):
    #    self.new_ans = self.cal_z(data, 0)

    #def cal_cross_entropy(self, ta):
    #    return ( ( np.dot( np.transpose(ta), np.log( np.clip( self.new_ans, 0.000000001, 1.5))) + np.dot( np.transpose( (1 - ta)), np.log( np.clip( (1 - self.new_ans), 0.000000001, 1.5)))) * -1)[0][0]

    def get_ans(self, data):
        temp1 = self.cal_z(data)
        temp = np.zeros(temp1.shape)
        for i in range(temp1.shape[0]):
            if temp1[i][0] >= 0:
                temp[i][0] = 1
            else:
                temp[i][0] = 0
        temp.astype(np.int64)
        return temp
