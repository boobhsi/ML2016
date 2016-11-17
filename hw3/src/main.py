from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np
import random
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

#testing_data = None
aul = None

al = pickle.load(open("data/all_label.p", "rb"))
#if int(argv[2])!=0: testing_data = pickle.load(open("data/test.p", "rb"))
if int(argv[2])!=0: auld = pickle.load(open("data/all_unlabel.p", "rb"))
class_count = len(al)
count_per_class = len(al[0])
count_for_ul = len(auld)

#print testing_data

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
    temp_ans = np.array([temp_ans])
    temp_ans_for_train = np.tile(temp_ans, (count_per_class-100, 1))
    temp_ans_for_vali = np.tile(temp_ans, (100, 1))
    if not isinstance(training_ans, np.ndarray):
        training_ans = temp_ans_for_train
        vali_ans = temp_ans_for_vali
    else:
        training_ans = np.concatenate((training_ans, temp_ans_for_train), axis=0)
        vali_ans = np.concatenate((vali_ans, temp_ans_for_vali), axis=0)

if int(argv[2])!=0:
    aul = np.array(auld).reshape(count_for_ul,  3, 32, 32)

training_data = np.array(training_data)
vali_data = np.array(vali_data)
print "There are {0} data for train, while {1} for validation".format(training_data.shape[0], vali_data.shape[0])
training_data = training_data.reshape(4000,3,32,32)
vali_data = vali_data.reshape(1000,3,32,32)
print "Check the shape of inputs: {0}".format(training_data.shape)
print "Check the shape of outputs: {0}".format(training_ans.shape)
print "Check the shape of validation inputs: {0}".format(vali_data.shape)
print "Check the shape of validation outputs: {0}".format(vali_ans.shape)

model.add(Convolution2D(20,3,3,input_shape=(3,32,32)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))
#model.add(Convolution2D(20,3,3))
#model.add(Activation("relu"))
#model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(50,3,3))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(output_dim=100))
model.add(Activation("sigmoid"))
model.add(Dropout(0.5))
#model.add(Dense(output_dim=100))
#model.add(Activation("relu"))
#model.add(Dropout(0.5))
#model.add(Dense(output_dim=60))
#model.add(Activation("relu"))
#model.add(Dropout(0.5))
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
rp = RMSprop(lr=0.0005)
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer=ao,
              metrics=["accuracy"])
datagen.fit(training_data)

cons_non_decay = 0
val_acc_max = -1000.
val_loss_min = 1000.
self_counter = 1
switch = 1
control = int(argv[3])

def fit_model(tx, ty):
    max_time = 1000
    global cons_non_decay
    global val_acc_max
    global val_loss_min
    global self_counter
    global switch
    global control
    while(max_time > 0):
        print "{0}th epoch, continuous overfitting: {1}".format(self_counter, cons_non_decay)
        print "current best val_loss_min = {0}".format(val_loss_min)
        #hist = model.fit_generator(datagen.flow(tx, ty, batch_size=int(argv[1])*3), nb_epoch=1, validation_data=(vali_data, vali_ans), samples_per_epoch=training_data.shape[0]*3)
        hist = model.fit(tx, ty, batch_size=int(argv[1]) * switch, nb_epoch=1, validation_data=(vali_data, vali_ans))
        print hist.history
        self_counter += 1
        if hist.history["val_loss"][0] < val_loss_min - 0.005:
            model.save(argv[4])
            cons_non_decay = 0
            val_loss_min = hist.history["val_loss"][0]
        else: cons_non_decay += 1
        if cons_non_decay == control and control != -1: break
        max_time -= 1
        if max_time == 0: break

if int(argv[2])==0:
    fit_model(training_data, training_ans)

elif int(argv[2])==1:
    fit_model(training_data, training_ans)
    switch *= 10
    while(1):
        judge = val_loss_min
        cons_non_decay = 0
        del model
        model = load_model(argv[4])
        print "model reloaded"
        new_ans = model.predict(aul)
        new_class = model.predict_classes(aul)
        new_l_counter = 0
        new_ty = []
        new_tx = []
        for i in range(aul.shape[0]):
            if random.random() < new_ans[i][new_class[i]] and new_ans[i][new_class[i]] > 0.5:
                temp_l = [0] * 10
                temp_l[new_class[i]] = 1
                new_ty = new_ty + [temp_l]
                new_tx = new_tx + [auld[i]]
                new_l_counter += 1
            #else:
            #    aul[i] = aul[i] * 0
            #temp_l[i][new_class[i]] = 1
        print "\nThere are {0} new data".format(new_l_counter)
        new_tx = np.array(new_tx)
        new_ty = np.array(new_ty)
        new_tx = new_tx.reshape(new_tx.shape[0], 3, 32, 32)
        #print new_tx.shape
        #print new_ty.shape
        new_tx = np.concatenate((training_data, new_tx), axis=0)
        new_ty = np.concatenate((training_ans, new_ty), axis=0)
        ao = Adam(lr=0.0005)
        model.compile(loss="categorical_crossentropy",
              optimizer=ao,
              metrics=["accuracy"])
        val_loss_min = 1000.
        fit_model(new_tx, new_ty)
        model.save(argv[4])

del model
model = load_model(argv[4])
td = pickle.load(open("data/test.p", "rb"))
testing_data =np.array( td["data"]).reshape(10000, 3, 32, 32)
class_ans = model.predict_classes(testing_data)

out_file = open(argv[5], "w")
out_file.write("ID,class\n")
for i in range(class_ans.shape[0]):
    out_file.write("{0},{1}\n".format(td["ID"][i], class_ans[i]))
