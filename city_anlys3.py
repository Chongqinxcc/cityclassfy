# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:04:54 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import os

'''
计算经纬度对应de网格，网格数量和大小后续进行调整。目前划分为等分的60*60个网格中
'''
def cal_mapIndex(dXMax1, dXMin1, dYMax1, dYMin1,lat_x,log_y):
    dXMax = dXMax1 
    dXMin = dXMin1
    dYMax = dYMax1
    dYMin = dYMin1

    m_nGridCount = 20
    m_dOriginX = dXMin
    m_dOriginY = dYMin
    dSizeX = (dXMax - dXMin) / m_nGridCount
    dSizeY = (dYMax-dYMin)/m_nGridCount
    #print("dSizeX:",dSizeY ,"dSizeY:",dSizeY)
          
    nXCol = int((lat_x - m_dOriginX) / dSizeX)
    nYCol = int((log_y - m_dOriginY) / dSizeY)
            
    iIndex = nXCol * m_nGridCount + nYCol

    return(iIndex)  
    
  

                         
'''
打开多个csv文件
'''                         
def openAllcsv_File(filepath):
    pathDir = os.listdir(filepath)  #获取当前路径下的文件名，返回List
    for s in pathDir:
        newDir = os.path.join(filepath,s)   #将文件命加入到当前文件路径后面
        if os.path.isfile(newDir):  #如果是文件
            if os.path.splitext(newDir)[1]==".xlsx": #判断是否是txt
                print(newDir)   #读文件进行处理
                read_cleanData(newDir)
                pass
        else:
            openAllcsv_File(newDir)   #如果不是文件，递归这个文件夹的路径
        
'''
读文件并进行处理，删除重复行，空白行进行处理等
'''  
def read_cleanData(filepath):
    
    nowDir = os.path.split(filepath)[0] #获取路径中的父文件夹路径
    sumDataDir = nowDir+r'\sumData.csv' #对新生成的文件进行命名的过程
    
    temp_Data = pd.read_excel(filepath,encoding='gb2312')
    
    #temp_Data = temp_Data.drop_duplicates(['城市','城市特点'])#去重
    
    print(temp_Data['城市'].value_counts())
    print(temp_Data.columns)
    
    dXMax, dXMin, dYMax, dYMin=temp_Data['经度'].max(),temp_Data['经度'].min(),\
                         temp_Data['纬度'].max(),temp_Data['纬度'].min()
                         
    #存放提取后的特征
    citylatlogDf = pd.DataFrame()
    #映射
    citylatlogDf = pd.concat([temp_Data['经度'],temp_Data['纬度']],axis=1)
    

    citylatlogDf['mapindex']=citylatlogDf.apply(lambda row:cal_mapIndex(dXMax, dXMin, dYMax, dYMin,row['经度'],
            row['纬度']),axis=1)

    temp_Data = pd.concat([temp_Data,citylatlogDf['mapindex']],axis=1,sort=True)
    print(temp_Data.dtypes)
    temp_Data['mapindex']=temp_Data['mapindex'].astype(str)
    temp_Data['邮编']=temp_Data['邮编'].astype(str)
    print('ddd:',temp_Data.dtypes)

    temp_Data.to_csv(sumDataDir,mode='a',index=False,header = True,encoding = 'gb2312')    
                
    
    

if __name__  == "__main__":
    
    #进行数据文件的预处理
    openAllcsv_File(r"C:\Users\Administrator\.spyder-py3\my_citydata")  
    
    filepath = r'C:\Users\Administrator\.spyder-py3\my_citydata'+r'\sumData.csv' 
    
    cityInfo_DF = pd.DataFrame()
    cityInfo_DF = pd.read_csv(filepath,encoding='gb2312')
    IsDuplicated = cityInfo_DF.duplicated()
    cityInfo_DF = cityInfo_DF.drop_duplicates()
    #cityInfo_DF.columns = ['日期', '时间', '城市', '邮编', '城市特点', '区域划分', '经度', '纬度','mapindex']
    print('数据类型',cityInfo_DF.dtypes)
    
    #testDF2_X = pd.DataFrame
    testDF=cityInfo_DF['城市']+' '+cityInfo_DF['城市特点']+' '+cityInfo_DF['邮编']+' '+cityInfo_DF['mapindex']
    testDF2_X=testDF.to_frame()
    testDF2_X.rename(columns={0:'联合数据'},inplace=True)

    #testDF2_y=pd.DataFrame
    testDF2_y= pd.concat([cityInfo_DF['区域划分']])
    testDF2_y=testDF2_y.to_frame()

    from sklearn.model_selection import  train_test_split
    X_train,X_test,y_train,y_test= train_test_split(testDF2_X,testDF2_y,random_state=42,test_size=0.25)
    
    '''
    导入词袋模型
    '''
    from sklearn.feature_extraction.text import  CountVectorizer
    vect=CountVectorizer()  # 实例化
    vect # 查看参数

    dir(vect) # 查看vect的属性 fit(.)函数从文档中学习出一个词汇表 transform(.)函数将指定文档转化为向量
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
    
    y_proba = pipe.predict_proba(X_test["联合数据"])
    
    city_id =X_test["联合数据"]
    
    test_id = y_test['区域划分']

    predDf = pd.DataFrame({'city_Id':city_id,'y_test':test_id,'区域':y_pred})
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
