 # -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:04:47 2019

@author: Administrator
"""

import numpy as np
import pandas as pd

#读取训练数据集
train = pd.read_csv('train.csv')
#读取测试数据集
test = pd.read_csv('test.csv')

print('训练数据集',train.shape,'测试数据集',test.shape)

#合并数据集，方便同时对两个数据集进行清洗
full = train.append(test,ignore_index=True)

print('合并后的数据集',full.shape)

#查看数据
des1=full.head()

des2=full.describe()

full.info()
'''
***************************************
年龄和船票价格存在缺失值，使用平均值填充的方法进行填充
***************************************
'''
full['Age']=full['Age'].fillna(full['Age'].mean())
full['Fare']=full['Fare'].fillna(full['Fare'].mean())

full['Cabin']=full['Cabin'].fillna('U')

from collections import Counter
print(Counter(full['Embarked']))

full['Embarked']=full['Embarked'].fillna('S')

full.info()

'''
***************************************
分类数据特征值提取：性别
男（male）对应数值 1
女（female)对应数值 0
***************************************
'''
sex_mapDict = {'male':1,'female':0}

#map函数：对于Series每个数据应用自定义函数计算
full['Sex'] = full['Sex'].map(sex_mapDict)
head = full.head()


'''
***************************************
分类数据特征值提取：登船港口
***************************************
'''
headE = full['Embarked'].head()
#存放提取后的特征
embarkdeDf = pd.DataFrame()
#使用get_dummies进行one-hot编码，列名前缀为Embarked
embarkedDf = pd.get_dummies(full['Embarked'],prefix='Embarked')
headE2 = embarkedDf.head()

#添加one-hot编码产生的虚拟变量（dummy variables)到泰坦尼克号数据集full
full = pd.concat([full,embarkedDf],axis=1)

#删除原来的Embarked
full.drop('Embarked',axis = 1,inplace = True)
headE3 = full.head()

'''
***************************************
提取客舱等级（Pclass）的特征值
***************************************
'''

#存放提取后的特征
pclassDf = pd.DataFrame()
#使用get_dummies进行one-hot编码，列名前缀为Pclass
pclassDf = pd.get_dummies(full['Pclass'],prefix='Pclass')
headc2 = pclassDf.head()

#添加one-hot编码产生的虚拟变量（dummy variables)到泰坦尼克号数据集full
full = pd.concat([full,pclassDf],axis=1)

#删除原来的Pclass
full.drop('Pclass',axis = 1,inplace = True)
headc3 = full.head()


'''
***************************************
从姓名（Name）中提取头衔的特征值
***************************************
'''
headName1 = full['Name'].head()



# 定义函数： 从姓名中获取头衔

# split()通过指定分隔符对字符串进行切片

def getTitle(name):           # Braund, Mr. Owen Harris

    str1 = name.split(',')[1] # Mr. Owen Harris

    str2 = str1.split('.')[0] # Mr

    # strip()方法用于移除字符串头尾指定的字符（默认为空格）

    str3 = str2.strip()
    return str3

# 存放提取后的特征

titleDf = pd.DataFrame()

# map函数：对于Series每个数据应用自定义函数计算

titleDf['Title'] = full['Name'].map(getTitle)

headname1 = titleDf.head()



# 姓名中头衔字符串与定义头衔类别的映射关系**

title_mapDict = {

    'Capt':        'Officer',

    'Col':         'Officer',

    'Major':       'Officer',

    'Jonkheer':    'Royalty',

    'Don':         'Royalty',

    'Sir':         'Royalty',

    'Dr':          'Officer',

    'Rev':         'Officer',

    'the Countess':'Royalty',

    'Dona':        'Royalty',

    'Mme':         'Mrs',

    'Mlle':        'Miss',

    'Ms':          'Mrs',

    'Mr':          'Mr',

    'Mrs':         'Mrs',

   'Miss':        'Miss',

    'Master':      'Master',
    'Lady':        'Royalty'
}

# map函数：对于Series每个数据应用自定义函数计算

titleDf['Title'] = titleDf['Title'].map(title_mapDict)

# 使用get_dummies进行one-hot编码

titleDf = pd.get_dummies(titleDf['Title'])

titleDf.head()



# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full

full = pd.concat([full,titleDf],axis=1)

# 删除姓名（Name）这一列

full.drop('Name',axis=1,inplace=True)

headname = full.head()

'''
***************************************
提取客舱号cabin
***************************************
'''
headcabin1 = full['Cabin'].head()

#存放客舱号信息
cabinDf = pd.DataFrame()

'''
客舱号的类别值是 首字母，例如：
C85类别映射为首字母C
'''
full['Cabin'] = full['Cabin'].map(lambda c:c[0]) #定义匿名函数，用于查找首字母
cabinvalue = full['Cabin'].value_counts()

#使用get_dummies进行one-hot编码，列名前缀为Cabin
cabinDf = pd.get_dummies(full['Cabin'],prefix = 'Cabin')
headcabin2 = cabinDf.head()

# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full

full = pd.concat([full,cabinDf],axis=1,sort=True)

# 删除姓名（Name）这一列

full.drop('Cabin',axis=1,inplace=True)

headname = full.head()

'''
***************************************
家庭类别
***************************************
'''


#存放家庭信息
familyDf = pd.DataFrame()

'''
家庭人数=同代直系亲属数（Parch）+不同代直系亲属数（SibSp）+乘客自己（因为乘客自己也是家庭成员的一个，所以这里加1）

小家庭Family_Single： 家庭人数=1

中等家庭Family_Small: 2<=家庭人数<=4

大家庭Family_Large: 家庭人数>=5

'''
familyDf['Familysize'] = full['Parch']+full['SibSp']+1


familyDf['family_singel']=familyDf['Familysize'].map(lambda s:1 if s==1 else 0)
familyDf['family_small']=familyDf['Familysize'].map(lambda s:1 if 2 <= s <=4 else 0)
familyDf['family_large']=familyDf['Familysize'].map(lambda s:1 if 5 <= s else 0)


familyDf.head()
# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full

full = pd.concat([full,familyDf],axis=1,sort=True)

headnaddfamily = full.head()


'''
***************************************
提取年龄（Age）的特征值
***************************************
'''
ageDf =pd.DataFrame()
'''
年龄类别：
儿童 0<年龄<=6
青少年 6<年龄<=18
青年 18<年龄<=40
中年 40<年龄<=60
老年 60<年龄
'''
ageDf['Child']       = full['Age'].map(lambda a:1 if 0<a<=6 else 0)
ageDf['Teenage']     = full['Age'].map(lambda a:1 if 6<a<=18 else 0)
ageDf['Youth']       = full['Age'].map(lambda a:1 if 18<a<=40 else 0)
ageDf['Middle_aged'] = full['Age'].map(lambda a:1 if 40<a<=60 else 0)
ageDf['Older']       = full['Age'].map(lambda a:1 if 60<a else 0)

ageDf.head()

full = pd.concat([full,ageDf],axis=1,sort=True)
full.head()


'''
***************************************
通过计算各个特征与标签的相关系数，进行特征选择，先求出相关性矩阵：
***************************************
'''
corrDf = full.corr()


'''
***************************************
查看各个特征与生成情况（Survived）的相关系数：
***************************************
'''
survived_corr=corrDf['Survived'].sort_values(ascending = False)


'''
***************************************
构建模型，根据相关系数的值选择特征做为模型的输入，titleDf,pclassDf，familyDf,
Fare,Sex,cabinDf,embarkedDf
建立训练数据集和测试数据集
***************************************
'''
full_X = pd.concat([titleDf,     # 头衔**

                    pclassDf,    # 客舱等级**
                    familyDf,    # 家庭大小**

                    full['Fare'],# 船票价格**
                    full['Sex'], # 性别**
                    cabinDf,     # 客舱号**

                    embarkedDf   # 登场港口**

                   ],axis=1,sort=True)

full_X.head()
 
sourceRow = 891

#原始数据集：特征
source_X = full_X.loc[0:sourceRow-1,:]
#原始数据集：标签
source_y = full.loc[0:sourceRow-1,'Survived']

#预测数据集：特征
pred_X = full_X.loc[sourceRow:,:]

print('原始数据集有多少行:',source_X.shape[0],'预测数据集有多少行:',pred_X.shape[0])

'''
从原始数据集（source）中拆分出训练数据集（记为train：用于模型训练）
和测试数据集（记为test：用于模型评估）：
'''
from sklearn.model_selection import train_test_split

train_X,test_X,train_y,test_y = train_test_split(source_X,source_y,test_size=0.8)


print('原始数据集特征:',source_X.shape,
      '训练数据集特征:',train_X.shape,
      '测试数据集特征:',test_X.shape)
     
print('原始数据集标签:',source_y.shape,
      '训练数据集标签:',train_y.shape,
      '测试数据集标签:',test_y.shape)

'''
选择一个机器学习算法，用于模型训练，这里选择逻辑回归（LOGISIC regression）
'''
     
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()


model.fit(train_X,train_y)


LogisticRegression(C=1.0,class_weight=None,dual = False,fit_intercept=True,
                   intercept_scaling=1,max_iter = 100,multi_class='ovr',n_jobs =1,
                   penalty = '12',random_state = None,solver = 'libinear',tol=0.0001,
                   verbose = 0,warm_start=False)

modelscore_LogisticRegression = model.score(test_X,test_y)

pred_y=model.predict(pred_X)

pred_y = pred_y.astype(int)

passenger_id = full.loc[sourceRow:,'PassengerId']

predDf = pd.DataFrame({'PassengerId':passenger_id,'Survived':pred_y})

predDf.shape
predDf.head()

predDf.to_csv('C:/Users/Administrator/.spyder-py3/titanic_pred.csv',index=False)































