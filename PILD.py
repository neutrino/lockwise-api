from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import Normalizer
from sklearn.externals import joblib
import numpy as np
from sys import argv


def PILD():
    testString = [argv[1]]
    ### If you use your codes to read the txt files into a list
    ### You can name the list 'testString'and start here
    ## Load all the models trained before
    modesSaved=joblib.load('processed/modes.sav')
    vectorizer = modesSaved[0]
    lsa = modesSaved[1]
    classifier = modesSaved[2]

    testMatrix = vectorizer.transform(testString) # string into tfidf matrix
    X_test = lsa.transform(testMatrix) # dimensionality reduction
    X_test = Normalizer(copy=False).transform(X_test) # normalization
    Y_predict = classifier.predict(X_test)

    testMatrix = vectorizer.transform(testString) # string into tfidf matrix
    X_test = lsa.transform(testMatrix) # dimensionality reduction
    X_test = Normalizer(copy=False).transform(X_test) # normalization
    Y_predict = classifier.predict(X_test) # predict with KNN method in our case
    #### Y_predict stores all the predicted values
    # print(testFileNames)
    # print(Y_predict)
    #############################################################
    ######### My own way to print the predicted values ##########
    print(Y_predict[0])


if __name__ == "__main__":
    PILD()


