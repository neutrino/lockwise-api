import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt

def readFilesintoList(folderName):
    files1 = [f for f in listdir(folderName) if isfile(join(folderName, f))]
    num1 = len(files1)
    stringAll = list()
    for i in range(num1):
        filei = folderName+files1[i]
        f = open(filei, 'r', encoding="utf-8")
        s0 = f.read()
        sepby = '- - - forwarded by'
        si = s0.split(sepby, 1)[0]+sepby
        stringAll.append(si)
    return stringAll

path0 = 'data/'
path1 = path0+'ham/'
path2 = path0+'spam/'

hamString = readFilesintoList(path1)
spamString = readFilesintoList(path2)
len1 = len(hamString) # 3672
len2 = len(spamString) #1500
lenAll = len1+len2
allString = hamString+spamString

def getAccuracy(allString):
    ratio = 0.7
    plist1= np.random.permutation(range(len1))
    plist2= np.random.permutation(range(len2))+len1
    train_id = np.append(plist1[0:int(ratio*len1)],plist2[0:int(ratio*len2)])
    test_id = np.append(plist1[int(ratio*len1):len1],plist2[int(ratio*len2):len2])

    #vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
    # the value in the matrix is the count of each words
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word',stop_words='english')
    # the value in the matrix is the tfidf of each word
    input_text=[]
    for i in train_id:
        input_text.append(allString[i])

    trainMatrix = vectorizer.fit_transform(input_text)
    #print(trainMatrix.shape)
    lsa = TruncatedSVD(200, algorithm = 'arpack')
    X_lsa = lsa.fit_transform(trainMatrix)
    X_lsa = Normalizer(copy=False).fit_transform(X_lsa)
    #print(X_lsa.shape)

    Y = np.ones(lenAll)
    Y[0:len1] = 0

    X_train = X_lsa
    Y_train = Y[train_id,]
    classifier = KNeighborsClassifier(n_neighbors=5) # KNN method
    #classifier = RandomForestClassifier(n_estimators=51) # random forest
    classifier.fit(X_train, Y_train)

    test_text=[]
    for i in test_id:
        test_text.append(allString[i])
    testMatrix = vectorizer.transform(test_text)
    X_test = lsa.transform(testMatrix)
    X_test = Normalizer(copy=False).transform(X_test)
    Y_test = Y[test_id,]
    Y_predict = classifier.predict(X_test)

    #modeSave = [vectorizer, lsa, classifier]
    #joblib.dump(modeSave, 'modes.sav')
    return [Y_predict, Y_test]

runTimes = 100
accAll = np.zeros(runTimes).reshape(1,-1)
accHamAll = np.zeros(runTimes).reshape(1,-1)
accSpamAll = np.zeros(runTimes).reshape(1,-1)

for j in range(1):
    [Y_predict, Y_test] = getAccuracy(allString)
    nTest = len(Y_test)
    accAll[j] = 1-sum(abs(Y_predict-Y_test))*1.0/nTest
    accHamAll[j] = 1-sum(Y_predict[Y_test==0])*1.0/sum(Y_test==0)
    accSpamAll[j] = sum(Y_predict[Y_test==1])*1.0/sum(Y_test==1)

print('The values given below is based on {} times tests.'.format(runTimes))
print('Accuracy: {0:.3f}.'.format(np.mean(accAll)))
print('Accuracy(Ham): {0:.3f}.'.format(np.mean(accHamAll)))
print('Accuracy(Spam): {0:.3f}.'.format(np.mean(accSpamAll)))
# print('Total Number of test: {}'.format(nTest))
# print('Accuracy: {0:.3f}.'.format(accuracy))
# print('Number of ham: {}'.format(sum(Y_test==0)))
# accHam = 1-sum(Y_predict[Y_test==0])*1.0/sum(Y_test==0)
# print('Ham Accuracy: {0:.3f}.'.format(accHam))
# accSpam = sum(Y_predict[Y_test==1])*1.0/sum(Y_test==1)
# print('Number of spam: {}'.format(sum(Y_test==1)))
# print('Spam Accuracy: {0:.3f}.'.format(accSpam))

