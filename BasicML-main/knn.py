#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:03:43 2020

@author: mahsa
KNN: Classification, Regression, Recommender system
"""

from collections import Counter
import math
import numpy as np
from numpy import linalg as LA

def knn(data,y,query):
    data_size = data.shape
    samples_num = data_size[0]
    dist = []
    for i in range (samples_num):
        data_sample = np.array(data)[i] #get the ith sample
        ds = LA.norm(data_sample-np.array(query),2)
        dist.append(ds)
        
    dist_sorted = np.sort(dist)#asecending    
    dist_index = np.argsort(dist)
    return dist_sorted,y[dist_index]

def Response(problem_flag,data,y,query,k):
    lst,ind = knn(data,y,query)
    k_lst , k_indx = lst[0:k] , ind[0:k]##choose the first k elements
    if problem_flag == 1:##Regression problem
        common_class = Counter(k_indx).most_common(1)[0]
        return common_class[0]

    else:
        mean_response = np.mean(k_indx)
        return  mean_response
         
    
if __name__ == '__main__':
    ## Regression Data
    
    #  height (inches)
    # weight (pounds)
    
    y = np.array([112.99, 136.49, 153.03, 142.34,144.30,123.30,141.49,136.46,112.37,127.45])
    reg_data = np.array([65.75, 71.52, 69.40, 68.22, 67.79, 68.70, 69.80, 70.01, 67.90, 66.49])
    query1 = np.array([60])
    problem_flag1 = 0
    k=3
    print('First Regression Result:',Response(problem_flag1,reg_data,y,query1,k))
    y = np.array([6,8,10,12,24,48,64])
    reg_data = np.array([3,4,5,6,12,24,32])
    query1 =np.array([3.5])
    print('Second Regression Result:',Response(problem_flag1,reg_data,y,query1,k))

    ## Classification Data
    #age
    #likes pineapple

    y = np.array([1,1,1,1,1,0,0,0,0,0])
    clf_data = np.array([22,23,21,18,19,25,27,29,31,45])
    query2 = np.array([33])
    problem_flag2 = 1
    print('First Classification Result:',Response(problem_flag2,clf_data,y,query2,k))