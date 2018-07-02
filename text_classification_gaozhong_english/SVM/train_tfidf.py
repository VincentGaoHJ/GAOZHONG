# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 17:16:30 2018

@author: Administrator
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV #它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.feature_selection import SelectKBest  
from sklearn.feature_selection import chi2  
from collections import defaultdict
from data_helper import Dataset
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
length = 250


#print(x_text)
print(len(x_text))

print("y.shape = {}".format(y.shape))


vectorizer = CountVectorizer(stop_words='english')
count = vectorizer.fit_transform(x_text)
print(vectorizer.get_feature_names())  
#print(vectorizer.vocabulary_)
print(count.toarray())
print(count.toarray().shape)

transformer = TfidfTransformer()
tfidf_matrix = transformer.fit_transform(count)
print(tfidf_matrix.toarray())
print(tfidf_matrix.toarray().shape)

x = tfidf_matrix.toarray()

# Load data
print("正在载入数据...")
# 函数dataset_read：输入文件名,返回训练集,测试集标签
# 注：embedding_w大小为vocabulary_size × embedding_size
train_data = Dataset(config.data_source, config.label_source)
train_data.dataset_read()
batch_train = train_data.next_batch()

print(x.shape)
print(batch_train[1].shape)

model1 = SelectKBest(chi2, k=100)#选择k个最佳特征  
model1.fit_transform(x, batch_train[1])#iris.data是特征数据，iris.target是标签数据，该函数可以选择出k个特征   

# Randomly shuffle data
np.random.seed(10) # 使得随机数列可预测
# 产生一个array：起始点0，结束点len(y)，步长1。然后将其打乱。
shuffle_indices = np.random.permutation(np.arange(len(batch_train[1]))) 
x_shuffled = x[shuffle_indices] # 将文件句子和标签以同样的方式打乱
y_shuffled = batch_train[1][shuffle_indices]


# Split train/test set
#直接把dev_sample_index前用于训练,dev_sample_index后的用于训练;
dev_sample_index = -1 * int(0.1 * float(len(batch_train[1]))) # -1:代表从后往前取
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]


TP=0    
TN=0
FP=0
FN=0

print(x_train.shape)
print(x_dev.shape)
print(y_train)
print(y_dev)

# Choose some parameter combinations to try
parameters = {'n_estimators':[1],
              'criterion':['entropy', 'gini']
              }
              #值为字典或者列表，即需要最优化的参数的取值，param_grid =param_test1，param_test1 = {'n_estimators':range(10,71,10)}。
# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)
        
clf = RandomForestClassifier()

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)#scoring准确度评价标准，默认None,这时需要使用score函数；或者如scoring='roc_auc'，根据所选模型不同，评价准则不同。字符串（函数名），或是可调用对象，需要其函数签名形如：scorer(estimator, X, y)；如果是None，则使用estimator的误差估计函数。
grid_obj = grid_obj.fit(x_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

clf = clf.fit(x_train, y_train)
test_predictions = clf.predict_proba(x_dev)
test_prediction = clf.predict(x_dev)
#print("测试集准确率:  %s " % accuracy_score(y_dev, test_predictions))

print(test_predictions[0])
print(len(test_predictions[0]))

print("开始评价")

scores = np.zeros((1760, 236))



for i in range(1760):
    for k in range(236):
#        print(i,k)
        scores[i][k] = 1 - test_predictions[k][i][0]
#        print(scores[i][k])
      
        
        
writeFile2 = open('.\\data\\gaozhong\\gaozhong_english\\test_data_english_eval.csv','a+', newline='') # 设置newline，否则两行之间会空一行
writer2 = csv.writer(writeFile2)
writer2.writerow(scores[0])
writeFile2.close()


pre, rec, pre_2, rec_2, pre_3, rec_3 = precision_recall(scores, y_dev)

print("pre_1 {}, rec_1 {:g}".format(pre, rec))
print("pre_2 {}, rec_2 {:g}".format(pre_2, rec_2))
print("pre_3 {}, rec_3 {:g}".format(pre_3, rec_3)) 

pre, rec, pre_2, rec_2, pre_3, rec_3 = precision_recall(test_prediction, y_dev)

print("pre_1 {}, rec_1 {:g}".format(pre, rec))
print("pre_2 {}, rec_2 {:g}".format(pre_2, rec_2))
print("pre_3 {}, rec_3 {:g}".format(pre_3, rec_3)) 
