# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:35:22 2019

@author: Administrator
"""

import numpy as np
'''进行数据处理 载入数据'''
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
    postingList = [x[0:4] for x in wordList]  #取前4列的内容
    return postingList , classVec  #postingList 文档集合 classvec 文档对应的分类

#不重复词（特征）列表
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

#输入词汇表 文档，输出文档向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print( "the word: %s is not in my Vocabulary!") % word
    return returnVec


'''多项式朴素贝叶斯函数的使用'''
from sklearn.naive_bayes import MultinomialNB

listOPosts,listClasses = loadDataSet('city1_3lei.txt')
myVocabList = createVocabList(listOPosts)
#构建训练集
trainMat=[]
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    
from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test = train_test_split(np.array(trainMat), np.array(listClasses), test_size=0.2, random_state=0)

clf = MultinomialNB(alpha=2.0)
clf.fit(X_train,Y_train)

y_predict = clf.predict(X_train)

print(clf.class_prior)  #各类的先验概率

print(clf.class_log_prior_) #各类标记的平滑先验概率对数值

from sklearn.metrics import classification_report
print('The accuracy of Navie Bayes Classifier is',clf.score(X_train,Y_train))
print(classification_report(Y_train,y_predict,target_names = ['东', '西','中']))

y_test_predict = clf.predict(x_test)
y_test_predict
y_test_predict == y_test

print('Misclassified samples: %d' % (y_test != y_test_predict).sum())
#Output:Misclassified samples: 3
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_test_predict))  #预测准确度,(len(y_test)-3)/len(y_test):0.9333333333333333
#Output:Accuracy: 0.93
'''
X_train = np.array(trainMat)
Y_train = np.array(listClasses)
clf = MultinomialNB(alpha=2.0)
clf.fit(X_train,Y_train)

y_predict = clf.predict(X_train)

print(clf.class_prior)  #各类的先验概率

print(clf.class_log_prior_) #各类标记的平滑先验概率对数值

from sklearn.metrics import classification_report
print('The accuracy of Navie Bayes Classifier is',clf.score(X_train,Y_train))
print(classification_report(Y_train,y_predict,target_names = ['东', '西']))

testEntry = ['11110000', '北京', '1级']
testEntry2 = ['88880000', '拉萨']
#testDoc =np.array(setOfWords2Vec(myVocabList, testEntry)).reshape(1,-1)
testDoc =np.array(setOfWords2Vec(myVocabList, testEntry2)).reshape(1,-1)
test_predict = clf.predict(testDoc)
'''