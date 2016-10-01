from datatype import *

class lr_agent:
    def __init__(self, rate, equ_coe):
        self.learning_rate = rate
        self.equ = linear_equ(17)
