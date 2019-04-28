# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:01:29 2019

@author: Administrator
"""

import pandas as pd
import os
import imp
#导入分解词
from sklearn.model_selection import train_test_split
#导入knn算法模型
from sklearn.neighbors import KNeighborsClassifier
# 导入分类器性能监测报告模块
from sklearn.metrics import classification_report


def loaddata(filepath):  #加载数据
    data=pd.read_csv(filepath)
    print('样本数据集:\n',data)
    #print('样本数据集:\n{0}'.format(data))

    # 步骤2：数据抽取
    # 获取war_count、love_count、movietype列数据
    data = data[['war_count', 'love_count', 'movietype']]
    print('原始样本数据集(数据抽取)：\n{0}'.format(data))

    # 返回数据
    return data


def splitdata(data):
    print('--数据划分--')
    X_train,X_test,y_train,y_test=train_test_split(data[['war_count','love_count']],data['movietype'],\
                                                   test_size=0.25,random_state=30)
    print('训练样本特征集:\n', X_train.values)
    print('训练样本标签集:\n', X_test.values)
    print('测试样本特征集:\n', y_train.values)
    print('测试样本标签集:\n', y_test.values)

    # 返回数据
    return X_train, X_test, y_train, y_test


def ModelTraing(X_train,X_test,y_train,y_yest):
    #先创建knn算法模型
    print('knn算法模型...')
    knn=KNeighborsClassifier(n_neighbors=3)

    #训练算法模型
    print('算法模型训练...')
    knn.fit(X_train,y_train)

    #训练模型评估
    result=knn.predict(X_test)
    print('knn训练模型测试报告:\n')
    print(classification_report(y_test,result,target_names=data['movietype'].unique()))

    return knn


if __name__=='__main__':
    # 设置数据文件的地址
    #filePath = os.getcwd() + '\data' + os.sep + 'movies.csv'
    filePath = os.getcwd() + os.sep + 'movies.csv'
    print(filePath)
    # 加载数据文件
    data = loaddata(filePath)
    # 数据划分
    X_train, X_test, y_train, y_test = splitdata(data)
    # 模型训练
    knn = ModelTraing(X_train, X_test, y_train, y_test)
    # 模型应用
    movietype = knn.predict([[20, 94]])
    print('电影分类预测结果为：{0}'.format(movietype[0]))