import numpy as np

class dataset:
    single_hour_count = 0
    average = None
    sdva = None
    size_fixed = False
    def __init__(self, data_shvar_count, data_hr_count):
        self.var_count = data_shvar_count * data_hr_count
        if dataset.size_fixed == False:
            dataset.average = np.array([0.0] * data_shvar_count)
            dataset.sdva = np.array([0.0] * data_shvar_count)
            dataset.size_fixed = True
        self.function_ans = []
        self.train_var = []
        self.name = []
        self.var_set = []
        self.temp_var_row = []

    def add_single_hour(self, inputpara, is_first, is_last):
        if is_last:
            self.train_var.append(inputpara[9])
        else:
            self.temp_var_row += inputpara
        if is_first:
            ia = np.array(inputpara)
            dataset.single_hour_count += 1
            dataset.average += ia
            dataset.sdva += ia ** 2

    def add_name(self, name):
        self.name.append(name)

    @staticmethod
    def get_global_sta():
        print dataset.single_hour_count
        dataset.average /= dataset.single_hour_count
        dataset.sdva = (dataset.sdva / dataset.single_hour_count - dataset.average ** 2) ** 0.5

    def normalize(self):
        a_temp = np.tile(dataset.average, 9)
        a_temp = np.repeat(a_temp[np.newaxis,:], self.var_set.shape[0], axis = 0)
        s_temp = np.tile(dataset.sdva, 9)
        s_temp = np.repeat(s_temp[np.newaxis,:], self.var_set.shape[0], axis = 0)
        self.var_set -= a_temp
        self.var_set /= s_temp

    def append(self):
        if type(self.var_set) is list:
            self.var_set = np.array(self.temp_var_row)
        else:
            self.var_set = np.vstack((self.var_set, self.temp_var_row))
        self.temp_var_row = []

    def __str__(self):

        return "{3} var_set:\n{0}\n{4} train_ans:\n{1}\nname:\n{2}".format(self.var_set, self.train_var, self.name, self.var_set.shape, len(self.train_var))

    def get_size(self):
        return self.var_set.shape[0]

    def get_var(self):
        return self.var_set

    def get_train_pm(self):
        return self.train_var

    def get_f_ans(self):
        return self.function_ans

    def arraize(self):
        if len(self.train_var) != 0:
            print "arraized!"
            self.train_var = np.transpose(np.array([self.train_var,]))

    def refresh_ans(self, equ):
        self.function_ans = equ.ans(self.get_var())

    def get_name(self):
        return self.name

class linear_equ:
    def __init__(self, order, coe, bias, max_order, data_count):
	self.coe = coe  * order
        self.bias = 0.0
        self.max_order = max_order
        self.data_count = data_count

    def ans(self, var):
        answer = np.dot(var, np.transpose(self.coe[:,:,0]))
        for i in range(1, self.max_order):
            answer += np.dot((var ** (i + 1)), np.transpose(self.coe[:,:,i]))
        return self.bias + answer

    def change_coe(self, new_coe, new_bias, order):
        self.coe = new_coe * order
        self.bias = new_bias

    def err_by_var(self, var, ta, data_count):
        return np.sum((ta - self.ans(var)) ** 2) / data_count ** 0.5

    def err_by_data(self, data):
        return (np.sum((data.get_train_pm() - data.get_f_ans()) ** 2) / data.get_size()) ** 0.5

    def err_pd_coe(self, data):
        temp = data.get_var()
        gra = np.dot(np.transpose((data.get_train_pm() - data.get_f_ans())[:, :]), temp)[:, :, np.newaxis]
        for i in range(1, self.max_order):
            gra = np.dstack((gra, (np.dot(np.transpose((data.get_train_pm() - data.get_f_ans())[:, :]), temp ** (i + 1))[:, :, np.newaxis])))
        return -2 * gra

    def err_pd_bias(self, data):
        return -2 * np.sum(data.get_train_pm() - data.get_f_ans())

    def __str__(self):
        return str(self.bias) + "\n" + str(self.coe)
