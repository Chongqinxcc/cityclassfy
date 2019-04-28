# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:53:59 2019

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''
cityList=[]
f=open('city_data2.txt','r',encoding='UTF-8')
for line in f.readlines():
    str1 = line.strip()
    str2 = str1.split('\t')
    cityList.append(str2)
    
   
del cityList[0]
postion = [x[1:3] for x in cityList] 
'''

'''
计算经纬度对应de网格，网格数量和大小后续进行调整。目前划分为等分的60*60个网格中
'''
def cal_mapIndex(lat_x,log_y):
    
    #将整个空间范围划分等份的网格
    dXMax = 180 
    dXMin = -180
    dYMax = 60
    dYMin = -60

    m_nGridCount = 60
    m_dOriginX = dXMin
    m_dOriginY = dYMin
    dSizeX = (dXMax - dXMin) / m_nGridCount
    dSizeY = (dYMax-dYMin)/m_nGridCount
    #print("dSizeX:",dSizeY ,"dSizeY:",dSizeY)
 
          
    nXCol = int((lat_x - m_dOriginX) / dSizeX)
    nYCol = int((log_y - m_dOriginY) / dSizeY)
            
    iIndex = nXCol * m_nGridCount + nYCol

    return(iIndex)  
    

#读city的经纬度
train = pd.read_csv('city3.csv',encoding='gb2312')
print('训练数据集',train.shape)

fullcity = train

'''
***************************************
将经纬度进行映射到地图ID
***************************************
'''
headE = fullcity['经度'].head()
#存放提取后的特征
citylatlogDf = pd.DataFrame()
#映射
citylatlogDf = pd.concat([fullcity['经度'],fullcity['纬度']],axis=1)
headE2 = citylatlogDf.head()

citylatlogDf['mapindex']=citylatlogDf.apply(lambda row:cal_mapIndex(row['经度'],
            row['纬度']),axis=1)

fullcity = pd.concat([fullcity,citylatlogDf['mapindex']],axis=1,sort=True)

map_list=citylatlogDf['mapindex'].tolist()
map_list2 = [str(i) for i in map_list]
fullcity['城市代码'].astype(str)
print(fullcity['城市代码'].dtype)

print(fullcity.info)
citynum_list = fullcity['城市代码'].tolist()
citynum_list2= [str(i) for i in citynum_list]



fullcity['城市代码2']=fullcity['城市代码'].astype(str)
print('类型:',fullcity['城市代码2'].dtype)

testDF2_X = pd.DataFrame
testDF=fullcity['城市级别']+' '+fullcity['城市特点']+' '+fullcity['城市代码2']
testDF2_X=testDF.to_frame()
testDF2_X.rename(columns={0:'联合数据'},inplace=True)

testDF2_y=pd.DataFrame
testDF2_y= pd.concat([fullcity['区域划分']])
testDF2_y=testDF2_y.to_frame()

from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test= train_test_split(testDF2_X,testDF2_y,random_state=42,test_size=0.25)

'''
导入词袋模型
'''
from sklearn.feature_extraction.text import  CountVectorizer
vect=CountVectorizer()  # 实例化
vect # 查看参数

dir(vect) # 查看vect的属性
'''
将分割后的文本进行fit_transform,进行向量化
'''
vect.fit_transform(X_train["联合数据"])
print('shape:',vect.fit_transform(X_train["联合数据"]).toarray().shape)

test3 = pd.DataFrame(vect.fit_transform(X_train["联合数据"]).toarray(),columns=vect.get_feature_names()).iloc[:,0:25].head()
print(vect.get_feature_names())

'''
模型构建
从sklearn 朴素贝叶斯中导入多维贝叶斯
朴素贝叶斯通常用来处理文本分类垃圾短信，速度飞快，效果一般都不会差很多
MultinomialNB类可以选择默认参数，如果模型预测能力不符合要求，可以适当调整
'''
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()  

from sklearn.pipeline import make_pipeline # 导入make_pipeline方法
pipe=make_pipeline(vect,nb)
print(pipe.steps) #  查看pipeline的步骤（与pipeline相似）

pipe.fit(X_train["联合数据"], y_train)

y_pred = pipe.predict(X_test["联合数据"]) 
# 对测试集进行预测（其中包括了转化以及预测）

# 模型对于测试集的准确率
from sklearn import  metrics
score1=metrics.accuracy_score(y_test,y_pred)

# 模型对于测试集的混淆矩阵
metrics.confusion_matrix(y_test,y_pred)
# 测试集中的预测结果：真阳性474个，假阳性112个，假阴性22个，真阴性为177个

'''
************************************
def get_confusion_matrix(conf,clas):
    import  matplotlib.pyplot as  plt
    fig,ax=plt.subplots(figsize=(2.5,2.5))
    ax.matshow(conf,cmap=plt.cm.Blues,alpha=0.3)
    tick_marks = np.arange(len(clas))
    plt.xticks(tick_marks,clas, rotation=45)
    plt.yticks(tick_marks, clas)
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(x=i,y=j,s=conf[i,j],
                   va='center',
                   ha='center')
    plt.xlabel("predict_label")
    plt.ylabel("true label")
    
    
conf=metrics.confusion_matrix(y_test,y_pred)
class_names=np.array(['0','1'])
get_confusion_matrix(np.array(conf),clas=class_names)
plt.show()
*****************************
'''
#对整个数据集进行预测分类
y_pred_all = pipe.predict(testDF2_X['联合数据'])
metrics.accuracy_score(testDF2_y,y_pred_all)
# 对于整个样本集的预测正确率，整个数据集的准确率高于测试集，说明有些过拟合

metrics.confusion_matrix(testDF2_y,y_pred_all)
#  真个数据集的混淆矩阵
'''
testDF2_y.value_counts()
# 初始样本中 正类与负类的数量

metrics.f1_score(y_true=testDF2_y,y_pred=y_pred_all)
# f1_score 评价模型对于真个数据集

metrics.recall_score(testDF2_y, y_pred_all)
# 检出率，也就是正类总样本检出的比例   真正/假阴+真正

metrics.precision_score(testDF2_y, y_pred_all)
#  准确率，  检测出的来正类中真正类的比例  真正/假阳+真正

print(metrics.classification_report(testDF2_y, y_pred_all))
# 分类报告
'''




#删除原来的Embarked
#full.drop('城市级别',axis = 1,inplace = True)
#headE3 = full.head()    




