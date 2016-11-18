from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import pickle
from sklearn.cluster import KMeans

al = pickle.load(open("data/all_label.p", "rb"))
td = pickle.load(open("data/test.p", "rb"))
auld = pickle.load(open("data/all_unlabel.p", "rb"))

data = td["data"]
for i in al:
    data = data + i
data = data + auld

encoding_dim = 256
input_img = Input(shape=(3*32*32,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(3*32*32, activation='sigmoid')(encoded)
autoencoder = Model(input=input_img, output=decoded)
encoder = Model(input=input_img, output=encoded)
x_train = np.array(data[0:len(data)-1000]).astype("float32") / 255.
x_test = np.array(data[len(data)-1000:len(data)]).astype("float32") / 255.

ao = Adam(lr=0.0005)
autoencoder.compile(optimizer=ao, loss="binary_crossentropy")
autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=200,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
kmeans = KMeans(n_clusters=10, random_state=0, max_iter=1000, n_init=20, n_jobs=-1).fit(encoded_imgs)

outfile = open(argv[1], "w")
outfile.write("ID,class\n")
for i in range(len(td["data"])):
    outfile.write("{0},{1}\n").format(td["ID"][i], kmeans.label_[i])
