# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:35:22 2019

@author: Administrator
"""

import numpy as np

def loadDataSet(infile):
    wordList=[]
    f=open(infile,'r')
    for line in f.readlines():
        str1 = line.strip('\n')
        str2 = str1.split('\t')
        wordList.append(str2)
    
    del wordList[0]
    classVec =[x[7] for x in wordList] 
    classVec = list(map(int, classVec))
    postingList = [x[0:4] for x in wordList]
    return postingList , classVec

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print( "the word: %s is not in my Vocabulary!") % word
    return returnVec
'''高斯贝叶斯函数的使用
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
#拟合数据
clf.fit(X, Y)

testEntry = [-0.8, -1]
print("预测的结果是：")
print(clf.predict([testEntry]))
print("属于每个类别的概率分别为：")
print(clf.predict_proba([testEntry]))
print("属于每个类别的对数转化后的概率分别为：")
print(clf.predict_log_proba([testEntry]))
'''

'''多项式朴素贝叶斯函数的使用'''
from sklearn.naive_bayes import MultinomialNB
#X = np.array([[1,2,3,4],[1,3,4,4],[2,4,5,5],[2,5,6,5],[3,4,5,6],[3,5,6,6]])
#y = np.array([1,1,4,2,3,3])
listOPosts,listClasses = loadDataSet('city1.txt')
myVocabList = createVocabList(listOPosts)
trainMat=[]
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

X = np.array(trainMat)
y = np.array(listClasses)
clf1 = MultinomialNB(alpha=2.0)
clf1.fit(X,y)
y_predict = clf1.predict(X)
print(clf1.class_log_prior_)
'''
from sklearn.metrics import classification_report
print('The accuracy of Navie Bayes Classifier is',clf1.score(X,y))
print(classification_report(y,y_predict,target_names = ['neg', 'pos']))
'''
testEntry = ['11110000', '北京', '1级']
testDoc =np.array(setOfWords2Vec(myVocabList, testEntry)).reshape(1,-1)
test_predict = clf1.predict(testDoc)
