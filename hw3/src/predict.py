from keras.models import load_model
import pickle
import numpy as np
from sys import argv

al = pickle.load(open(argv[1] + "test.p", "rb"))
model = load_model(argv[2])
testing_data = np.array(al["data"]).reshape(10000, 3, 32, 32)

class_ans = model.predict_classes(testing_data)

outfile = open(argv[3], "w")
outfile.write("ID,class\n")
for i in range(class_ans.shape[0]):
    outfile.write("{0},{1}\n".format(al["ID"][i], class_ans[i]))
