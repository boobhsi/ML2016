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
            #print "train_ans_insert! <- {0}".format(inputpara[9])
            self.train_var.append(inputpara[9])
        else:
            #print "train_var_insert <- {0}".format(inputpara)
            self.temp_var_row += inputpara
            #print "current row : {0}".format(self.temp_var_row)
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
        #print dataset.average
        #print dataset.sdva

    def normalize(self):
        a_temp = np.tile(dataset.average, 9)
        #a_temp = np.concatenate((a_temp, np.zeros(1, 18)), axis = 1)
        a_temp = np.repeat(a_temp[np.newaxis,:], self.var_set.shape[0], axis = 0)
        s_temp = np.tile(dataset.sdva, 9)
        #s_temp = np.concatenate((s_temp, np.ones(1, 18)), axis = 1)
        s_temp = np.repeat(s_temp[np.newaxis,:], self.var_set.shape[0], axis = 0)
        #print a_temp.shape
        #print s_temp.shape
        #print self.var_set.shape
        self.var_set -= a_temp
        self.var_set /= s_temp
        #print self.var_set.shape

    def append(self):
        #print len(self.temp_var_row)
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
            #print self.train_var
        #print self.train_var.shape

    def refresh_ans(self, equ):
        self.function_ans = equ.ans(self.get_var())
        #print self.function_ans

    def get_name(self):
        return self.name
