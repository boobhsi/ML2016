import numpy as np

class linear_equ:
    def __init__(self, order, coe, bias):
        #self.varc = varc
        self.order = order
        self.coe = coe
        self.bias = 0.0
        #self.acc_order = [0] * varc
        #acc_temp = 1
        #for i in range(len(order)):
        #    self.acc_order[i] = acc_temp
        #    acc_temp += order[i]
        #self.seve = acc_temp
        #self.order_mal = [-1] * len(coe)
        #var_index = 0
        #coe_index = 1
        #while coe_index < len(coe):
        #    for i in range(order[var_index]):
        #        self.order_mal[coe_index + i] = var_index
        #    coe_index += order[var_index]
        #    var_index += 1
        #print self.acc_order
        #print self.order_mal

    def ans(self, var):
        #answer = self.coe[0]
        #coe_index = 1
        #for var_num in range(self.varc):
        #    for order_num in range(self.order[var_num]):
        #        answer += self.coe[coe_index] * pow(var[var_num], order_num + 1)
        #        coe_index += 1
        #return answer
        #print self.coe.shape
        #print var.shape
        #print self.order.shape
        return self.bias + np.sum(np.power(var, self.order) * self.coe)

    def change_coe(self, new_coe, new_bias):
        self.coe = new_coe
        self.bias = new_bias

    def err_by_var(self, var, ta):
        return (ta - self.ans(var)) ** 2

    def err_by_data(self, data):
        return (data.get_f_pm() - data.function_ans) ** 2

    def err_pd_coe(self, data):
        #var_index = self.order_mal[coe_index]
        #if var_index != -1:
            #print data.get_f_pm()
            #print data.function_ans
            #print var_index
            #print data.get_var()[var_index]
            #print self.acc_order[var_index]
            #err =  2 * (data.get_f_pm() - data.function_ans) * -1 * pow(data.get_var()[var_index], coe_index + 1 - self.acc_order[var_index])
            #print err
            #return err
        #else:
            #return 2 * (data.get_f_pm() - data.function_ans) * -1
        return -2 * (data.get_f_pm() - data.function_ans) * np.power(data.get_var(), self.order)

    def err_pd_bias(self, data):
        return -2 * (data.get_f_pm() - data.function_ans)

    def print_coe(self):
        print self.bias
        print self.coe

class designed_equ_1:
    def __init__(self, varc, order, coe, days, weight):
        self.days = days
        self.varc = varc
        self.order = order
        self.weight = weight
        self.coe = coe
        self.acc_order = [0] * varc
        acc_temp = 1
        for i in range(len(order)):
            self.acc_order[i] = acc_temp
            acc_temp += order[i]
        self.seve = acc_temp
        self.order_mal = [-1] * len(coe)
        var_index = 0
        coe_index = 1
        while coe_index < len(coe):
            for i in range(order[var_index]):
                self.order_mal[coe_index + i] = var_index
            coe_index += order[var_index]
            var_index += 1
        #print self.acc_order
        #print self.order_mal

    def ans(self, var):
        answer = self.weight[0]
        for day_num in range(self.days):
            temp = self.coe[0]
            coe_index = 1
            for var_num in range(self.varc):
                for order_num in range(self.order[var_num]):
                    temp += self.coe[coe_index] * pow(var[var_num + day_num * self.varc], order_num + 1)
                    coe_index += 1
            answer += temp * self.weight[day_num + 1]
        return answer

    def change_coe(self, new_coe):
        self.coe = new_coe

    def change_weight(self, new_weight):
        self.weight = new_weight

    def err_by_var(self, var, ta):
        return (ta - self.ans(var)) ** 2

    def err_by_data(self, data):
        return (data.get_f_pm() - data.function_ans) ** 2

    def err_pd_coe(self, coe_index, data):
        var_index = self.order_mal[coe_index]
        if var_index != -1:
            err = 0.0
            #print data.get_f_pm()
            #print data.function_ans
            #print var_index
            #print data.get_var()[var_index]
            #print self.acc_order[var_index]
            for day_num in range(self.days):
                err += self.weight[day_num + 1] * pow(data.get_var()[var_index + day_num * self.varc], coe_index + 1 - self.acc_order[var_index])
            #print err
            return err * -2.0* (data.get_f_pm() - data.function_ans)
        else:
            return -2.0 * (data.get_f_pm() - data.function_ans) * sum(self.weight[1:10])

    def err_pd_weight(self, weight_index, data):
        if weight_index == 0:
            return -2.0 * (data.get_f_pm() - data.function_ans)
        else:
            ans_od = self.coe[0]
            coe_index = 1
            for var_num in range(self.varc):
                for order_num in range(self.order[var_num]):
                    ans_od += self.coe[coe_index] * pow(data.get_var()[var_num + (weight_index - 1) * self.varc], order_num + 1)
                    coe_index += 1
            return -2.0 * (data.get_f_pm() - data.function_ans) * ans_od

    def print_coe(self):
        print [round(i, 6) for i in self.coe]

    def print_weight(self):
        print [round(i, 6) for i in self.weight]

class obv_data:
    id = 0
    single_hour_count = 0
    para_count = 18
    var_count = 9 * 18
    average_ft = np.array([0.0] * 18)
    sd_ft = np.array([0.0] * 18)

    def __init__(self, name = None):
        self.id = obv_data.id
        self.th_para = []
        obv_data.id += 1
        self.function_ans = 0.0
        self.name = name

    def add_oh_para(self, inputpara, is_first):
        self.th_para.append(inputpara)
        ia = np.array(inputpara)
        if is_first:
            obv_data.single_hour_count += 1
            obv_data.average_ft += ia
            obv_data.sd_ft += ia ** 2

    def get_global_sta(self):
        obv_data.average_ft /= obv_data.single_hour_count
        obv_data.sd_ft = ((obv_data.sd_ft / obv_data.single_hour_count) - obv_data.average_ft ** 2) ** 0.5

    def normalize(self):
        a_temp = np.repeat(obv_data.average_ft[np.newaxis,:], 10, 0)
        #print a_temp.shape
        s_temp = np.repeat(obv_data.sd_ft[np.newaxis], 10, 0)
        self.arraize_para()
        #print self.th_para.shape
        self.th_para /= s_temp
        self.th_para -= (a_temp / s_temp)

    def arraize_para(self):
        self.th_para = np.array(self.th_para)

    def __str__(self):
        temp = "data_id:" + str(self.id) + " ->\n"
        for i in self.th_para:
            temp += (str(i) + "\n")
        return temp

    def get_var(self):
        return self.th_para[range(9)]

    def get_f_pm(self):
        return self.th_para[9][9]

    def set_f_pm(self, equ):
        p_para = [0.0] * 18
        self.refresh_ans(equ)
        p_para[9] = self.function_ans
        np.vstack((self.th_para, np.array(p_para)))

    def refresh_ans(self, equ):
        self.function_ans = equ.ans(self.get_var())
