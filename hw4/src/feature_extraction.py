from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sys import argv
import csv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sys import argv
import operator
import codecs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

trans = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), lowercase=True, stop_words="english", max_df=0.95, min_df=2)


#counts = CountVectorizer(ngram_range=(1, 3), stop_words="english", analyzer="word")
#trans = TfidfVectorizer(use_idf=False)

dd = codecs.open(argv[1] + "docs.txt", "r", "utf-8")

#dds = dd.read()
#dd.close()
#dds = re.compile("\w+").findall(dds)

#print dds

#tokens = nltk.word_tokenize(dds)

#stemmer = nltk.stem.snowball.SnowballStemmer("english")

#dds = "".join([stemmer.stem(i) + " " for i in dds if not i.isdigit()])

#print dds

print "Fitting TfV..."
#voc = trans.fit_transform(dd)
voc = trans.fit_transform(dd)
#check = [(0, 0.0)] * 70000
#value = voc.data
#column_index = voc.indices
#row_pointers = voc.indptr

#for i in range(len(value)):
#    co = column_index[i]
#    if value[i] != check[co][1]: check[co] = (co, value[i])
#print check
#sorted_check = sorted(check, key=lambda tup: tup[1])
#print sorted_check
dd.close()

#one = []

#for i in sorted_check:
#    if i[1] == 1.0: one.append(i[0])

#for k, v in trans.vocabulary_.iteritems():
#    if v in one: print k

#print trans.vocabulary_
#print trans.get_params()

#print sorted_dic
data = [i.strip() for i in open(argv[1] + "title_StackOverflow.txt", "r")]

#print "Transforming training data to vectors"
#train_data = trans.transform(data)
train_data = trans.transform(data)
print train_data.shape

svd = TruncatedSVD(n_components=20)
normalizer = Normalizer(copy=False, norm="l2")
#normalizer2 = Normalizer(copy=False, norm="l1")
lsa = make_pipeline(svd, normalizer)

#print "Reducing dimension of vectors..."
trai_data = lsa.fit_transform(train_data)
#train_data = svd.fit_transform(train_data)
#print train_data.shape

coorize = TruncatedSVD(n_components=3)
psa = make_pipeline(coorize, normalizer)
coor = psa.fit_transform(trai_data)

cluster_count = int(argv[3])

#clf = SpectralClustering(n_clusters=20)
#clf = AgglomerativeClustering(n_clusters=20)
clf = KMeans(n_clusters=cluster_count)

#print "Clustering and predicting..."
predicted = clf.fit_predict(trai_data)
"""
print "Verifing data..."
veri = [0] * cluster_count
for i in clf.labels_:
    veri[int(i)] += 1
print veri
mv = max(veri)
maxI = 0
"""
"""
for i in range(len(veri)):
    if veri[i] == mv: maxI = i
for i in range(len(train_data)):
    if clf.labels_[i] == maxI: print data[i]
"""

#print "Opening testing data..."
outfile = open(argv[2], "w")
outfile.write("ID,Ans\n")

not_first = False

#print "Judging answers..."

with open(argv[1] + "check_index.csv", "rb") as check_csv:
    check_data = csv.reader(check_csv, delimiter=",", quotechar='"')
    for i in check_data:
        if not not_first: not_first = True
        else:
            outfile.write("{0},{1}\n".format(i[0], 1 if predicted[int(i[1])] == predicted[int(i[2])] else 0))

#print coor
#xs = [x[0] for x in coor]
#ys = [x[1] for x in coor]

colors = cm.rainbow(np.linspace(0, 1, cluster_count))
#print colors
counter = 0

for xy in coor:
    plt.scatter(xy[0], xy[2], color=colors[predicted[counter]])
    counter += 1
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.savefig("exp/my_" + argv[3] + "_cluster.png")
plt.close()


###ans = np.load("data/real_label.npy")
"""
#print len(data)

list_labeled = ([],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[])
#print list_labeled

for i in range(len(data)):
    #print data[i]
    #print ans[i]-1
    list_labeled[ans[i]-1].append(data[i])

ofile = open("obser_idfFal.csv", "w")

for i in list_labeled:
    #print i
    #print len(i)
    analysis = TfidfVectorizer(use_idf=False, max_features=10)
    analysis.fit(i)
    #print "vocabulary:"
    a =  analysis.vocabulary_

    a = sorted(a.items(), key=operator.itemgetter(1), reverse=True)

    for i in range(len(a)):
        if i == 9: ofile.write("{0}\n".format(a[i][0].decode("utf-8").encode("ascii")))
        else: ofile.write("{0},".format(a[i][0].decode("utf-8").encode("ascii")))

    #print "stopwords:"
    #print analysis.stop_words_
"""
"""
ans_colors = cm.rainbow(np.linspace(0, 1, 20))
counter = 0

for xy in coor:
    plt.scatter(xy[0], xy[2], color=ans_colors[ans[counter]-1])
    counter += 1
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.savefig("exp/labeled.png")
"""
