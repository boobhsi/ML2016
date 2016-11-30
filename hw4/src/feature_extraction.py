from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sys import argv
import csv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import nltk

trans = TfidfVectorizer(stop_words="english", analyzer="word", ngram_range=(1, 3), lowercase=True, max_df=0.5, min_df=2)

dd = open("data/docs.txt", "r")
print "Fitting TfV..."
trans.fit(dd)
#print trans.vocabulary_
dd.close()

data = [i.strip() for i in open("data/title_StackOverflow.txt", "r")]

print "Transforming training data to vectors"
train_data = trans.transform(data)
print train_data.shape

svd = TruncatedSVD(n_components=100)
normalizer = Normalizer(copy=False, norm="l2")
lsa = make_pipeline(svd, normalizer)

print "Reducing dimension of vectors..."
train_data = lsa.fit_transform(train_data)
print train_data.shape

cluster_count = 70

#clf = SpectralClustering(n_clusters=20)
#clf = AgglomerativeClustering(n_clusters=20)
clf = KMeans(n_clusters=cluster_count)

print "Clustering and predicting..."
predicted = clf.fit_predict(train_data)
print "Verifing data..."
veri = [0] * cluster_count
for i in clf.labels_:
    veri[int(i)] += 1
print veri
mv = max(veri)
maxI = 0
"""
for i in range(len(veri)):
    if veri[i] == mv: maxI = i
for i in range(len(train_data)):
    if clf.labels_[i] == maxI: print data[i]
"""
print "Opening testing data..."
outfile = open(argv[1], "w")
outfile.write("ID,Ans\n")

not_first = False

print "Judging answers..."
with open("data/check_index.csv", "rb") as check_csv:
    check_data = csv.reader(check_csv, delimiter=",", quotechar='"')
    for i in check_data:
        if not not_first: not_first = True
        else:
            outfile.write("{0},{1}\n".format(i[0], 1 if predicted[int(i[1])] == predicted[int(i[2])] else 0))
