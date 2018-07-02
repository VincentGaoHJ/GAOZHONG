# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 21:55:34 2018

@author: gaoha
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV #它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。
from sklearn.metrics import accuracy_score,make_scorer
from data_helper import Dataset
from collections import defaultdict
import datetime
from config import config
import numpy as np
import csv

def load_data_and_labels(data_file, labels_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    x_text = []
    y = []
    
    with open(data_file, encoding = "utf-8") as csvFile:
        readCSV = csv.reader(csvFile, delimiter = ",")
        for row in readCSV:
            row = "".join(row)
            x_text.append(row)    
    
    with open(labels_file, encoding = "utf-8") as csvFile2:
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
            
#            print(len(row))
            y.append(row)
            



  
    print("x = {}".format(len(x_text)))
    print("y = {}".format(len(y)))
           
    return x_text, y



def precision_recall(scores, y_batch):        
    y_indices = scores.argsort()[:, -1:][:, ::-1]
    pre = 0.0
    rec = 0.0
    for i in range(len(y_batch)):
        intersec_true = 0
        for j in y_indices[i]:
            intersec_true += y_batch[i][j]
        true_total_count = np.count_nonzero(y_batch[i] == 1)
        pred_total_count = len(y_indices[i])
        pre += intersec_true*1.0/pred_total_count
        rec += intersec_true*1.0/true_total_count
    pre = pre/len(y_batch)
    rec = rec/len(y_batch)
    
    
    
    y_indices_2 = scores.argsort()[:, -2:][:, ::-1]
    pre_2 = 0.0
    rec_2 = 0.0
    for i in range(len(y_batch)):
        intersec_true = 0
        for j in y_indices_2[i]:
            intersec_true += y_batch[i][j]
        true_total_count = np.count_nonzero(y_batch[i] == 1)
        pred_total_count = len(y_indices_2[i])
        pre_2 += intersec_true*1.0/pred_total_count
        rec_2 += intersec_true*1.0/true_total_count
    pre_2 = pre_2/len(y_batch)
    rec_2 = rec_2/len(y_batch)
    
    
    y_indices_3 = scores.argsort()[:, -3:][:, ::-1]
    pre_3 = 0.0
    rec_3 = 0.0
    for i in range(len(y_batch)):
        intersec_true = 0
        for j in y_indices_3[i]:
            intersec_true += y_batch[i][j]
        true_total_count = np.count_nonzero(y_batch[i] == 1)
        pred_total_count = len(y_indices_3[i])
        pre_3 += intersec_true*1.0/pred_total_count
        rec_3 += intersec_true*1.0/true_total_count
    pre_3 = pre_3/len(y_batch)
    rec_3 = rec_3/len(y_batch)
    
    return pre, rec, pre_2, rec_2, pre_3, rec_3


x_text, y = load_data_and_labels(config.data_source, config.label_source)
y = np.array(y)



time_str = datetime.datetime.now().isoformat()
print(time_str)

length = 250

a = np.zeros(shape=(len(y),length,128))
b = []

W2V = {}

W2V_file = csv.reader(open(".\data\gaozhong\gaozhong_english\W2V.csv","r",encoding = "utf-8"))

h = 1
for stu in W2V_file:
    W2V[stu[1]] = stu[2]
#    if h % 1000 == 0 :
#        print(stu[1])
#        print(stu[2])
    h += 1

m = 0
for x in x_text:
    if m % 1000 == 1:
        print(m)
    li = x.split(" ")
    k = 0
    for i in li:
#        print(i)
        if k == length-1 :
            break
        i = i.encode('utf-8').decode('utf-8-sig')
        a[m][k] = W2V[i].split(",")
        k += 1     
    m += 1

a = np.mean(a, axis=1)



print(a.shape)
print("a")

# Randomly shuffle data
np.random.seed(10) # 使得随机数列可预测
# 产生一个array：起始点0，结束点len(y)，步长1。然后将其打乱。
shuffle_indices = np.random.permutation(np.arange(len(a))) 
x_shuffled = a[shuffle_indices] # 将文件句子和标签以同样的方式打乱
y_shuffled = y[shuffle_indices]


# Split train/test set
#直接把dev_sample_index前用于训练,dev_sample_index后的用于训练;
dev_sample_index = -1 * int(0.1 * float(len(y))) # -1:代表从后往前取
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

time_str = datetime.datetime.now().isoformat()
print(time_str)

TP=0    
TN=0
FP=0
FN=0

print(x_train.shape)
print(x_dev.shape)
print(y_train)
print(y_dev)

# Choose some parameter combinations to try
parameters = {'n_estimators':[20],
              'criterion':['entropy', 'gini']
              }
              #值为字典或者列表，即需要最优化的参数的取值，param_grid =param_test1，param_test1 = {'n_estimators':range(10,71,10)}。
# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)
        
clf = RandomForestClassifier()

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)#scoring准确度评价标准，默认None,这时需要使用score函数；或者如scoring='roc_auc'，根据所选模型不同，评价准则不同。字符串（函数名），或是可调用对象，需要其函数签名形如：scorer(estimator, X, y)；如果是None，则使用estimator的误差估计函数。
grid_obj = grid_obj.fit(x_train, y_train)

print(grid_obj.best_estimator_.get_params())
time_str = datetime.datetime.now().isoformat()
print()

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_


# =============================================================================
# for h in range(length):
#     print(h)
#         
#     x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
#     y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
#     
#     clf = clf.fit(x_train, y_train)
#     test_predictions = clf.predict_proba(x_shuffled)
#     
#     #print("测试集准确率:  %s " % accuracy_score(y_dev, test_predictions))
#     
#     print(test_predictions[0])
#     print(len(test_predictions[0]))
#     
#     time_str = datetime.datetime.now().isoformat()
#     print(time_str)
#     
#     
#     print("开始评价")
#     
#     scores = np.zeros((len(test_predictions[0]), 236))
#     
# #    new_feature = [0]*len(test_predictions[0])
#     
#     for i in range(len(test_predictions[0])):
#         for k in range(236):
#     #        print(i,k)
#             scores[i][k] = 1 - test_predictions[k][i][0]
# #            if scores[i][k] == 1:
# #                new_feature[i] = k
#     #        print(scores[i][k])
#           
#     if h != 0 :
#         x_shuffled = x_shuffled[ : , 236: ]
#         x_shuffled = np.c_[scores,x_shuffled]
#         print(x_shuffled.shape)
#     else:
#         x_shuffled = np.c_[scores,x_shuffled]
#         print(x_shuffled.shape)
# =============================================================================
        
        
        
#print(h)
    
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

clf = clf.fit(x_train, y_train)
test_predictions = clf.predict_proba(x_dev)

#print("测试集准确率:  %s " % accuracy_score(y_dev, test_predictions))

print(test_predictions[0])
print(len(test_predictions[0]))

time_str = datetime.datetime.now().isoformat()
print(time_str)


print("开始评价")

scores = np.zeros((len(test_predictions[0]), 236))     

for i in range(len(test_predictions[0])):
     for k in range(236):
         scores[i][k] = 1 - test_predictions[k][i][0]
       
        
        
        
# =============================================================================
#             
#     writeFile2 = open('.\\data\\gaozhong\\gaozhong_english\\test_data_english_eval.csv','a+', newline='') # 设置newline，否则两行之间会空一行
#     writer2 = csv.writer(writeFile2)
#     writer2.writerow(scores[0])
#     writeFile2.close()
#     
# =============================================================================
    
pre, rec, pre_2, rec_2, pre_3, rec_3 = precision_recall(scores, y_dev)

print("pre_1 {}, rec_1 {:g}".format(pre, rec))
print("pre_2 {}, rec_2 {:g}".format(pre_2, rec_2))
print("pre_3 {}, rec_3 {:g}".format(pre_3, rec_3)) 


writeFile2 = open('.\\data\\gaozhong\\gaozhong_english\\dev_data_english_eval.csv','a+', newline='') # 设置newline，否则两行之间会空一行
writer2 = csv.writer(writeFile2)
writer2.writerow([h, pre, rec, pre_2, rec_2, pre_3, rec_3])
writeFile2.close()
