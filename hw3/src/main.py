from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np
from sys import argv

early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=0.000001)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")

model = Sequential()

al = pickle.load(open("data/all_label.p", "rb"))
class_count = len(al)
count_per_class = len(al[0])

#print al

vali_data = []
vali_ans = 0
training_ans = 0
training_data = []

for i in range(class_count):
    temp_ans = [0] * class_count
    temp_ans[i] = 1
    temp = al[i]
    ran_list = np.random.choice(range(count_per_class), 100, replace=False)
    for j in range(len(temp)):
        if j not in ran_list: training_data += [temp[j]]
        else: vali_data += [temp[j]]
    #for j in temp:
    #    rgb_graph = [[[[0] for x in range(32)] for y in range(32)] for z in range(3)]
    #    if check == 0: print rgb_graph
    #    for k in range(3):
    #        for m in range(32):
    #            for l in range(32):
    #                #print k*1024+m*32+l
    #                rgb_graph[k][m][l] = j[k*1024+m*32+l]
    #            if check == 0: print rgb_graph[k][m]
    #    if check == 0:
    #        print j
    #        print rgb_graph[0]
    #        check += 1
    #    if not isinstance(training_data, list):
    #        training_data = [rgb_graph]
    #    else:
    #        training_data.append(rgb_graph)
    #print temp.shape
    #print temp_ans
    temp_ans = np.array([temp_ans])
    #print temp_ans
    temp_ans_for_train = np.tile(temp_ans, (count_per_class-100, 1))
    temp_ans_for_vali = np.tile(temp_ans, (100, 1))
    #print temp_ans.shape
    if not isinstance(training_ans, np.ndarray):
        training_ans = temp_ans_for_train
        vali_ans = temp_ans_for_vali
    else:
        training_ans = np.concatenate((training_ans, temp_ans_for_train), axis=0)
        vali_ans = np.concatenate((vali_ans, temp_ans_for_vali), axis=0)
#training_data = np.array(training_data)
training_data = np.array(training_data)
vali_data = np.array(vali_data)
print "There are {0} data for train, while {1} for validation".format(training_data.shape[0], vali_data.shape[0])
training_data = training_data.reshape(4000,3,32,32)
vali_data = vali_data.reshape(1000,3,32,32)
#print test.shape
#test = test.reshape(5000,3,32,32)
#print test
print "Check the shape of inputs: {0}".format(training_data.shape)
print "Check the shape of outputs: {0}".format(training_ans.shape)
print "Check the shape of validation inputs: {0}".format(vali_data.shape)
print "Check the shape of validation outputs: {0}".format(vali_ans.shape)
#print test
#print training_ans

model.add(Convolution2D(20,3,3,input_shape=(3,32,32)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(20,3,3))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(50,3,3))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(output_dim=200))
model.add(Activation("relu"))
model.add(Dropout(0.3))
#model.add(Dense(output_dim=200))
#model.add(Activation("relu"))
#model.add(Dropout(0.3))
#model.add(Dense(output_dim=200))
#model.add(Activation("relu"))
#model.add(Dropout(0.3))
#model.add(Dense(output_dim=200))
#model.add(Activation("relu"))
#model.add(Dropout(0.3))
#model.add(Dense(output_dim=200))
#model.add(Activation("relu"))
#model.add(Dropout(0.3))

"""
model.add(Dense(output_dim=500))
model.add(Activation("relu"))
"""
"""
model.add(Dense(output_dim=500))
model.add(Activation("relu"))
model.add(Dense(output_dim=200))
model.add(Activation("relu"))
model.add(Dense(output_dim=100))
model.add(Activation("relu"))
"""

ao = Adam(lr=0.0001)
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer=ao,
              metrics=["accuracy"])
datagen.fit(training_data)

self_counter = 1
val_loss_min = 1000.0
val_acc_max = -1000.0
cons_non_decay = 0
while(1):
    print "{0}th epoch, continuous overfitting: {1}".format(self_counter, cons_non_decay)
    print "current best val_acc = {0}".format(val_acc_max)
    hist = model.fit_generator(datagen.flow(training_data, training_ans, batch_size=int(argv[1])*5), nb_epoch=1, validation_data=(vali_data, vali_ans), samples_per_epoch=training_data.shape[0]*5)
    print hist.history
    self_counter += 1
    if hist.history["val_acc"][0] > val_acc_max:
        model.save(argv[2])
        cons_non_decay = 0
        val_acc_max = hist.history["val_acc"][0]
    else: cons_non_decay += 1
    if cons_non_decay == 40: break
