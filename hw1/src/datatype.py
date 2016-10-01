impotrt

class obv_data:
    def __init__(self):
        self.parameters = []
        self.pm = 0.0
        self.para_count = 17
    def add_para(self, inputpara):
        self.parameters.append(inputpara)
    def set_pm(self, inputpara):
        self.pm = inputpara


class linear_equ:
    def __init__(self, varc, order, coe):
        self.varc = varc
        self.order = order
        self.seve = sum(self.order) + 1
        self.coe = coe

    def ans(self, var, pdiff):
        d_coe = []
        if pdiff == 0:
            d_coe = self.coe
        else:
            d_coe = [0] * seve
            pdiff_var = 1 + sum(order[0:pdiff - 1])
            d_coe[0] = self.coe[pdiff_var]
            for order_index in range(pdiff_var + 1, pdiff_var + order[diff]):
                order_ac = order_index - pdiff_var + 1
                d_coe[order_index] = self.coe[order_index] * order_ac
        answer = d_coe[0]
        coe_index = 1
        for var_num in range(self.varc):
            for order_num in range(self.order[var_num]):
                answer = d_coe[coe_index] * pow(var[var_num], order_num + 1)
        return answer

    def change_coe(new_coe):
        self.coe = new_coe
