# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:15:46 2019

@author: Administrator
"""
import pandas as pd
import numpy as np

df = pd.DataFrame([[np.nan,2,np.nan,0],
                   [3,4,np.nan,1],
                   [np.nan,np.nan,np.nan,5],
                   [np.nan,3,np.nan,4]],
                   columns=list('ABCD'))

df['C']=df['C'].fillna(df['B'])

city_data=pd.read_excel('city_test1.xlsx',encoding='gb2312')

print(city_data.columns)

IsDuplicated = city_data.duplicated(['城市','城市特点'])

city_data = city_data.drop_duplicates(['城市','城市特点'])

cityname = pd.Series(city_data['城市'])

print(cityname.unique())

print(cityname.value_counts())

print(pd.value_counts(city_data['城市'],sort=True))

print(city_data['城市'].value_counts())

print(city_data['经度'].max())

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
    

    
dXMax, dXMin, dYMax, dYMin=city_data['经度'].max(),city_data['经度'].min(),\
                         city_data['纬度'].max(),city_data['纬度'].min()
                         
'''
***************************************
将经纬度进行映射到地图ID
***************************************
'''
headE = city_data['经度'].head()
#存放提取后的特征
citylatlogDf = pd.DataFrame()
#映射
citylatlogDf = pd.concat([city_data['经度'],city_data['纬度']],axis=1)
headE2 = citylatlogDf.head()

citylatlogDf['mapindex']=citylatlogDf.apply(lambda row:cal_mapIndex(dXMax, dXMin, dYMax, dYMin,row['经度'],
            row['纬度']),axis=1)

city_data = pd.concat([city_data,citylatlogDf['mapindex']],axis=1,sort=True)

map_list=citylatlogDf['mapindex'].tolist()
map_list2 = [str(i) for i in map_list]

                       
