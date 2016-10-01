class obv_data:
    def __init__(self):
        self.parameters = []
        self.pm = 0.0
        self.para_count = 17
    def add_para(self, inputpara):
        self.parameters.append(inputpara)
    def set_pm(self, inputpara):
        self.pm = inputpara
