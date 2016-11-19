import pickle
import numpy as np
from sys import argv

al = pickle.load(open(argv[1] + "test.p", "rb"))
model = pickle.load(open(argv[2], "rb"))


outfile = open(argv[3], "w")
outfile.write("ID,class\n")
for i in range(len(model)):
    outfile.write("{0},{1}\n".format(al["ID"][i], model[i]))
