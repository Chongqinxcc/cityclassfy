# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:20:12 2019

@author: Administrator
"""

#!/usr/bin/env python
#coding = utf-8
'''
Author: buffer
E-mail: 179770346@qq.com
desc:   此处使用网格索引算法查找，网格划分越细，查找效率越高，可能内存会越大，需要权衡.
        还可以使用四叉树算法来做空间索引，但是考虑四叉树可能稍微复杂，使用网格算法
'''
 
#空间对象
class GeoObject:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
 
#网格索引查找类
class GeoSearch:
    def __init__(self):
        try:
            file = open("city_data2.txt", 'r',encoding='UTF-8')
        except IOError:
            print("Can't open the file!")
            
        #读取空间数据到内存中,并获取整个空间数据的外接矩形范围
        objArray = []
        dXMax = -999.0 #初始化一个无穷小的范围
        dXMin = 999.0
        dYMax = -999.0
        dYMin = 999.0
        
        file.readline();
        for cursor in file:
            info = []
            for str in cursor.split("\t"):
                if str == "" or str == " ":
                    continue
                info.append(str)
            len1=(len(info))
            if len(info) < 3:
                #print info[0], info[1], info[2], info[3], len(info)
                continue
            mydXMax = mydXMin = float(info[2]) 
            mydYMax = mydYMin = float(info[1])
            objArray.append(GeoObject(info[0], mydXMax, mydYMax))
            if dXMax < mydXMax:
                dXMax = mydXMax
            if dXMin > mydXMin:
               dXMin = mydXMin
            if dYMax < mydYMax:
                dYMax = mydYMax
            if dYMin > mydYMin:
                dYMin = mydYMin
        print( dXMax, dXMin, dYMax, dYMin)
            
        #将整个空间范围划分等份的网格
        self.m_nGridCount = 32
        self.m_dOriginX = dXMin
        self.m_dOriginY = dYMin
        self.dSizeX = (dXMax - dXMin) / self.m_nGridCount
        print(self.dSizeX)
        self.dSizeY = (dYMax - dYMin) / self.m_nGridCount
        print(self.dSizeY)
        self.m_vIndexCells = []
        for i in range(0, self.m_nGridCount * self.m_nGridCount + 1):
            self.m_vIndexCells.append([])
        #self.m_vIndexCells = [list()] * (self.m_nGridCount * self.m_nGridCount)
        print( self.m_nGridCount, self.m_dOriginX, self.m_dOriginY, \
              self.dSizeX, self.dSizeY, len(self.m_vIndexCells), len(objArray))
 
        #将每个对象注册到空间索引中
        for obj in objArray:
            nXCol = int((obj.x - self.m_dOriginX) / self.dSizeX)
            nYCol = int((obj.y - self.m_dOriginY) / self.dSizeY)
            if nXCol < 0:
                nXCol = 0
            if nXCol >= self.m_nGridCount:
                nXCol = self.m_nGridCount - 1
            if nYCol < 0:
                nYCol = 0
            if nYCol <= self.m_nGridCount:
                nYCol = self.m_nGridCount - 1
                
            iIndex = nXCol * self.m_nGridCount + nYCol
            self.m_vIndexCells[iIndex].append(obj)
            #print iIndex, len(self.m_vIndexCells[iIndex])
            
    def Search(self, dXMax, dXMin, dYMax, dYMin):
        #计算输入外接矩形所在网格的范围
        nXMin = int((dXMin - self.m_dOriginX) / self.dSizeX)
        nXMax = int((dXMax - self.m_dOriginX) / self.dSizeX + 0.5) #四舍五入
        nYMin = int((dYMin - self.m_dOriginY) / self.dSizeY)
        nYMax = int((dYMax - self.m_dOriginY) / self.dSizeY + 0.5)
        print( nXMin, nXMax, nYMin, nYMax)
        
        #遍历这些网格得到所有符合条件的设备
        for n in range(nYMin, nYMax):
            for m in range(nXMin, nXMax):
                iIndex = n * self.m_nGridCount + m
                for obj in self.m_vIndexCells[iIndex]:
                    if obj.x <= dXMax and obj.x >= dXMin and \
                       obj.y <= dYMax and obj.y >= dYMin:
                        print( obj.name)
        
 
if __name__ == '__main__':
    geoSearch = GeoSearch();
    geoSearch.Search(134.28, 99.25, 52.97, 34.25)

