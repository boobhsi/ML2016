import numpy as np

class neuron:
    def __init__(self, fc):
        self.fc = fc
        self.omega = np.zeros((fc, 1))
        self.bias = 0.0
        self.new_ans = None

    def cal_z(self, data):
        return np.dot(data, self.omega) + self.bias

    def cal_pro(self, data):
        return 1 / (1 + np.exp((self.cal_z(data) * -1)))

    def refresh_para(self, data, learning_rate, ta):
        self.omega = self.omega + learning_rate * np.dot(np.transpose(data), (ta - self.new_ans))
        self.bias = self.bias + learning_rate * np.sum((ta - self.new_ans))
        self.new_ans = self.cal_pro(data)

    def init_ans(self, data):
        self.new_ans = self.cal_pro(data)

    def cal_cross_entropy(self, ta):
        return ( ( np.dot( np.transpose(ta), np.log( np.clip( self.new_ans, 0.000000001, 1.5))) + np.dot( np.transpose( (1 - ta)), np.log( np.clip( (1 - self.new_ans), 0.000000001, 1.5)))) * -1)[0][0]

    def get_ans(self, data):
        temp1 = self.cal_pro(data)
        temp = np.zeros(temp1.shape)
        for i in range(temp1.shape[0]):
            if temp1[i][0] >= 0.5:
                temp[i][0] = 1
            else:
                temp[i][0] = 0
        temp.astype(np.int64)
        return temp
