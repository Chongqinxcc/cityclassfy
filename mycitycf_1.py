# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:20:08 2019

@author: Administrator
"""

import pandas as pd

#读取训练数据集
train = pd.read_csv('city2.csv',encoding='gb2312')
print('训练数据集',train.shape)

full = train

'''
***************************************
分类数据特征值提取：城市级别
***************************************
'''
headE = full['城市级别'].head()
#存放提取后的特征
citylevelDf = pd.DataFrame()
#使用get_dummies进行one-hot编码，列名前缀为Embarked
citylevelDf = pd.get_dummies(full['城市级别'],prefix='城市级别')
headE2 = citylevelDf.head()

#添加one-hot编码产生的虚拟变量（dummy variables)到泰坦尼克号数据集full
full = pd.concat([full,citylevelDf],axis=1)

#删除原来的Embarked
full.drop('城市级别',axis = 1,inplace = True)
headE3 = full.head()

