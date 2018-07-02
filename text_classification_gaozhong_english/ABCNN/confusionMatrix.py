# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 18:17:21 2018

@author: gaoha
"""

import numpy as np
import pandas as pd
from config import config
from data_helper import Dataset
import csv
from collections import defaultdict

def dev_batch_iter(data):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/64) + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * 64
        end_index = min((batch_num + 1) * 64, data_size)
        yield data[start_index:end_index]
        
        
label = []       
chinese_sequence = [] 
with open(config.label_source, encoding = "utf-8") as csvFile2:
    readCSV = csv.reader(csvFile2, delimiter = ",")
    for row in readCSV:
        d = defaultdict(list)
        for k,va in [(v,i) for i,v in enumerate(row)]:
            d[k].append(va)
    
        for k in range(len(d.get("1.0"))):
            index = d.get("1.0")[k]
            row[index] = 1
        for k in range(len(d.get("0.0"))):
            index = d.get("0.0")[k]
            row[index] = 0
        
        label.append(row)
        

        
with open(config.chinese_sequence_9, encoding = "utf-8") as csvFile:
    readCSV = csv.reader(csvFile, delimiter = ",")
    for row in readCSV:
#                row = "".join(row)
        chinese_sequence = row


chinese_sequence = list(map(int, chinese_sequence))
print(chinese_sequence)

label = np.array(label, dtype='float32')

label = label[chinese_sequence]

y_pred = np.loadtxt(open(".\\data\\gaozhong\\gaozhong_chinese\\test_data_chinese_eval_scores.csv","rb"),delimiter=",",skiprows=0)  

print(y_pred)

ConfusionMatrix = np.zeros((config.nums_classes,config.nums_classes))
ConfusionMatrix_1 = np.zeros((config.nums_classes,config.nums_classes))
ConfusionMatrix_2 = np.zeros((config.nums_classes,config.nums_classes))
ConfusionMatrix_3 = np.zeros((config.nums_classes,config.nums_classes))

print(ConfusionMatrix.shape)



test_batches = dev_batch_iter(list(zip(y_pred, label)))

for test_batch in test_batches:
    y_pred, y_true = zip(*test_batch)
    y_pred = list(y_pred)
    y_true = list(y_true)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    k = 0
    for i in range(len(y_true)):
        position = np.nonzero(y_true[i])
        k = len(position[0])
        y_indices = y_pred[i].argsort()[-k:][::-1]
        print(y_indices)
        print(position[0])
        if k == 1 :
            ConfusionMatrix[int(y_indices)][int(position[0])] += 1
            ConfusionMatrix_1[int(y_indices)][int(position[0])] += 1
        else :
            s = set(list(y_indices)).intersection(set(list(position[0])))
            print(s)
            for tureNum in s:
                ConfusionMatrix[tureNum][tureNum] += 1
                if k == 2 :
                    ConfusionMatrix_2[tureNum][tureNum] += 1
                if k == 3 :
                    ConfusionMatrix_3[tureNum][tureNum] += 1
            if k - len(s) != 0 :
                predict = set(list(y_indices)).difference(set(list(position[0])))  
                true = set(list(position[0])).difference(set(list(y_indices)))
                print(true)
                print(predict)
                for predictNum in predict:
                    for trueNum in true:
                        ConfusionMatrix[predictNum][trueNum] += 1/(len(true))
                        if k == 2 :
                            ConfusionMatrix_2[predictNum][tureNum] += 1
                        if k == 3 :
                            ConfusionMatrix_3[predictNum][tureNum] += 1

        print("===========")
print(ConfusionMatrix)
np.savetxt('data/gaozhong/gaozhong_chinese/ConfusionMatrix.csv',ConfusionMatrix , delimiter = ",")
np.savetxt('data/gaozhong/gaozhong_chinese/ConfusionMatrix_1.csv',ConfusionMatrix_1 , delimiter = ",")
np.savetxt('data/gaozhong/gaozhong_chinese/ConfusionMatrix_2.csv',ConfusionMatrix_2 , delimiter = ",")
np.savetxt('data/gaozhong/gaozhong_chinese/ConfusionMatrix_3.csv',ConfusionMatrix_3 , delimiter = ",")
        
        
        

