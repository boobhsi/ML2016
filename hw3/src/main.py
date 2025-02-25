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
from keras.layers.normalization import BatchNormalization

#define data generator
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")

model = Sequential()
td = None #testing data
aul = None #all unlabeled data
auld = None #all 
count_for_ul = 0 #total count of unlabeled data

print "Loading labeled data...."
al = pickle.load(open(argv[5] + "all_label.p", "rb"))
if int(argv[2])!=0:
	print "Loading testing data...."
	td = pickle.load(open(argv[5] + "test.p", "rb"))
	print "Loading unlabeled data...."
	auld = pickle.load(open(argv[5] + "all_unlabel.p", "rb"))
	auld = auld + td["data"]
        count_for_ul = len(auld)

class_count = len(al)
count_per_class = len(al[0])

vali_data = []
vali_ans = 0
training_ans = 0
training_data = []

print "Generating data for training and validation...."
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
    aul = np.array(auld).reshape(count_for_ul,  3, 32, 32) #define array of all unlabeled data as aul
training_data = np.array(training_data)
vali_data = np.array(vali_data)

print "There are {0} data for train, while {1} for validation".format(training_data.shape[0], vali_data.shape[0])
training_data = training_data.reshape(4000,3,32,32)
vali_data = vali_data.reshape(1000,3,32,32)
#print "Check the shape of inputs: {0}".format(training_data.shape)
#print "Check the shape of outputs: {0}".format(training_ans.shape)
#print "Check the shape of validation inputs: {0}".format(vali_data.shape)
#print "Check the shape of validation outputs: {0}".format(vali_ans.shape)

model.add(Convolution2D(20,3,3,input_shape=(3,32,32),dim_ordering="th"))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2),dim_ordering="th"))
model.add(Convolution2D(50,3,3,dim_ordering="th"))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2),dim_ordering="th"))
model.add(Convolution2D(100,3,3,dim_ordering="th"))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2),dim_ordering="th"))
model.add(Flatten())
model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=-1, momentum=0.99, weights=None, beta_init="zero", gamma_init="one", gamma_regularizer=None, beta_regularizer=None))
model.add(Dense(output_dim=200))
model.add(Activation("sigmoid"))
model.add(Dropout(0.1))

ao = Adam(lr=0.0001)
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer=ao,
              metrics=["accuracy"])

datagen.fit(training_data)

cons_non_decay = 0 #self-defined earlystopping counter
val_acc_max = -1000.
val_loss_min = 1000.
self_counter = 1
switch = 1 #variable for batch_size
sw = 1 #variable for data generation
first_time = 0 #the least times model must be trained
control = int(argv[3]) #">=0" -> self-defined es patience, "==-1" -> model will be trained until max_time==0
#"==-2" -> make sure model will be trained by max_per times
max_time = 200
max_per = 10

def fit_model(tx, ty):
    global first_time
    global max_time
    global cons_non_decay
    global val_acc_max
    global val_loss_min
    global self_counter
    global switch
    global control
    global sw
    global max_per
    while(max_time > 0):
        print "{0}th epoch, continuous overfitting: {1}".format(self_counter, cons_non_decay)
        print "current best val_loss_min = {0}".format(val_loss_min)
        hist = model.fit_generator(datagen.flow(tx, ty, batch_size=int(argv[1])*sw*switch), nb_epoch=1, validation_data=(vali_data, vali_ans), samples_per_epoch=training_data.shape[0]*sw)
        #hist = model.fit(tx, ty, batch_size=int(argv[1]) * switch, nb_epoch=1, validation_data=(vali_data, vali_ans))
        print hist.history
        self_counter += 1
	if control == -2:
	    model.save(argv[4])
	    val_loss_min = hist.history["val_loss"][0]
	    val_acc_max = hist.history["val_acc"][0]
        elif hist.history["val_loss"][0] < val_loss_min - 0.005 and first_time == 0:
            model.save(argv[4])
            cons_non_decay = 0
            val_loss_min = hist.history["val_loss"][0]
	    val_acc_max = hist.history["val_acc"][0]
        else:
            if first_time == 0: cons_non_decay += 1
	max_time -= 1
        if cons_non_decay == control and control >= 0: break
        if first_time > 0: first_time -= 1
        if control == -2:
	    if max_per == 1: break
	    else: max_per -= 1

if int(argv[2])==0:
    sw = 5
    fit_model(training_data, training_ans)

elif int(argv[2])==1:
    sw = 5
    control = 10
    max_per = 20
    fit_model(training_data, training_ans)
    while(aul.shape[0] > 1000 and max_time > 0):
	#first_time = 5
	#cons_non_decay = 0
	#switch = 1
        #fit_model(training_data, training_ans)
        #judge = val_loss_min
        control = -2
        max_per = 20
        sw = 1
        cons_non_decay = 0
	print "model reloading"
        model = load_model(argv[4])
        print "model reloaded"
        new_ans = model.predict(aul)
        new_class = new_ans.argmax(axis=1)
        new_l_counter = 0
	tm = new_ans.max(axis=1)
	#print tm.shape
	a = (((np.sum(new_ans ** 2, axis=1) - (tm ** 2) + (1 - tm) ** 2) / 10) ** 0.5) < 0.1
	#print a
	b = (np.random.random_sample(tm.shape) < (val_acc_max))
	#print b
	cd = np.logical_and(a, b)
	#print cd
	index = np.nonzero(cd)
	#print index
	new_tx = aul[index]
	new_ty = np.identity(10, dtype=int)[new_class[index]]
	#print new_ans[index]
	#print new_ty
	aul = np.delete(aul, index, axis=0)
        #print "\nThere are {0} new data".format(new_tx.shape[0])
        new_tx = new_tx.reshape(new_tx.shape[0], 3, 32, 32)
        #print new_tx.shape
        #print new_ty.shape
        training_data = np.concatenate((training_data, new_tx), axis=0)
        training_ans = np.concatenate((training_ans, new_ty), axis=0)
        #ao = Adam(lr=0.0005)
        #model.compile(loss="categorical_crossentropy",
        #      optimizer=ao,
        #      metrics=["accuracy"])
        val_loss_min = 1000.
	#switch *= training_data.shape[0] / 4000
	#model.reset_states()
        fit_model(training_data, training_ans)
        #model.save(argv[4])

"""
model.load_weights(argv[4])
if int(argv[2]) == 0: td = pickle.load(open("data/test.p", "rb"))
testing_data =np.array( td["data"]).reshape(len(td["data"]), 3, 32, 32)
class_ans = model.predict_classes(testing_data)

out_file = open(argv[5], "w")
out_file.write("ID,class\n")
for i in range(class_ans.shape[0]):
    out_file.write("{0},{1}\n".format(td["ID"][i], class_ans[i]))
"""
