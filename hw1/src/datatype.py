class linear_equ:

    def __init__(self, varc, order, coe):
        self.varc = varc
        self.order = order
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
        answer = self.coe[0]
        coe_index = 1
        for var_num in range(self.varc):
            for order_num in range(self.order[var_num]):
                answer += self.coe[coe_index] * pow(var[var_num], order_num + 1)
                coe_index += 1
        return answer

    def change_coe(self, new_coe):
        self.coe = new_coe

    def err_by_var(self, var, ta):
        return (ta - self.ans(var)) ** 2

    def err_by_data(self, data):
        return (data.get_f_pm() - data.function_ans) ** 2

    def err_pd_coe(self, coe_index, data):
        var_index = self.order_mal[coe_index]
        if var_index != -1:
            #print data.get_f_pm()
            #print data.function_ans
            #print var_index
            #print data.get_var()[var_index]
            #print self.acc_order[var_index]
            err =  2 * (data.get_f_pm() - data.function_ans) * -1 * pow(data.get_var()[var_index], coe_index + 1 - self.acc_order[var_index])
            #print err
            return err
        else:
            return 2 * (data.get_f_pm() - data.function_ans) * -1

    def print_coe(self):
        print [round(i, 4) for i in self.coe]

class obv_data:
    id = 0
    para_count = 18
    var_count = 9*18
    average_ft = [0.0] * 18
    sd_ft = [0.0] * 18

    def __init__(self, name = None):
        self.id = obv_data.id
        self.th_para = []
        obv_data.id += 1
        self.variables = []
        self.function_ans = 0.0
        self.name = name

    def add_oh_para(self, inputpara):
        self.th_para.append(inputpara)
        for i in range(len(inputpara)):
            obv_data.average_ft[i] += inputpara[i]
            obv_data.sd_ft[i] += inputpara[i] ** 2

    def normalize(self):
        for i in range(obv_data.para_count):
            obv_data.average_ft[i] /= (obv_data.id * 10)
            obv_data.sd_ft[i] = (obv_data.sd_ft[i] / (obv_data.id * 10) -
                                 (obv_data.average_ft[i] ** 2)) ** 0.5

    def __str__(self):
        temp = "data_id:" + str(self.id) + " ->\n"
        for i in range(len(self.th_para)):
            temp = temp + str(self.th_para[i]) + "\n"
        return temp

    def get_var(self):
        new_var = [0.0] * obv_data.var_count
        for i in range(9):
            for j in range(self.para_count):
                #new_var[i * self.para_count + j] = (self.th_para[i][j] - obv_data.average_ft[j]) / obv_data.sd_ft[j]
                new_var[i * self.para_count + j] = self.th_para[i][j]
        return new_var

    def get_f_pm(self):
        return self.th_para[9][9]

    def set_f_pm(self, equ):
        p_para = [0.0] * 18
        self.refresh_ans(equ)
        p_para[9] = self.function_ans
        self.th_para.append(p_para)

    def refresh_ans(self, equ):
        self.function_ans = equ.ans(self.get_var())
