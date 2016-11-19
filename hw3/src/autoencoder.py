from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import pickle
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sys import argv

es = EarlyStopping(monitor="val_loss", patience=2, min_delta=0.001)

al = pickle.load(open(argv[1] + "all_label.p", "rb"))
td = pickle.load(open(argv[1] + "test.p", "rb"))
auld = pickle.load(open(argv[1] + "all_unlabel.p", "rb"))

data = td["data"]
data = data + auld
for i in al:
    data = data + i

encoding_dim = 256
input_img = Input(shape=(3*32*32,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(3*32*32, activation='sigmoid')(encoded)
autoencoder = Model(input=input_img, output=decoded)
encoder = Model(input=input_img, output=encoded)
x_train = np.array(data[0:len(data)-1000]).astype("float32") / 255.
x_vali = np.array(data[len(data)-1000:len(data)]).astype("float32") / 255.

ao = Adam(lr=0.001)
autoencoder.compile(optimizer=ao, loss="binary_crossentropy")
autoencoder.fit(x_train, x_train,
                nb_epoch=1,
                batch_size=200,
                validation_data=(x_vali, x_vali))

encoded_imgs = encoder.predict(np.array(data))
clf = NearestCentroid()
y = np.repeat(np.array([0,1,2,3,4,5,6,7,8,9]), 500)
clf.fit(encoded_imgs[55000:60000], y)
ans = clf.predict(encoded_imgs[0:10000])
with open(argv[2], "wb") as output:
    pickle.dump(ans, output, pickle.HIGHEST_PROTOCOL)

"""
metric = np.zeros((10, 10))
for i in range(5000):
	metric[kmeans.labels_[i+45000+10000]][i/500] += 1
max_in = np.argmax(metric, axis=1)
"""
"""
outfile = open(argv[1], "w")
outfile.write("ID,class\n")
for i in range(len(td["data"])):
    outfile.write("{0},{1}\n".format(td["ID"][i], ans[i]))
    """
