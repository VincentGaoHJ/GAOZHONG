# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:28:26 2018
@author: qizhiliu
"""
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV #它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。
from sklearn.metrics import accuracy_score,make_scorer

train_data = pd.read_csv(r'.\data\train_usual.csv')
test_data = pd.read_csv(r'.\data\test_usual.csv')

#将数据转化未label（0-N）形式
def encode_features(df_train, df_test):
    features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()#标准化标签，将标签值统一转换成range(标签值个数-1)范围内
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    
    return df_train, df_test
TP=0    
TN=0
FP=0
FN=0
train_data, test_data= encode_features(train_data, test_data)
print(train_data)
print(test_data)
print(train_data.shape)
print(test_data.shape)

X_all = train_data.drop(['Class'], axis=1)
X_all = X_all.drop(['Amount'], axis=1)
y_all = train_data['Class']
test_pr=test_data.drop(['Class'],axis=1)
test_pr=test_pr.drop(['Amount'],axis=1)
y_result = test_data['Class']


num_test = 0.10
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=3)
print(X_train.shape)
print(X_test.shape)
print(y_train)
print(y_test)

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
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

clf = clf.fit(X_train, y_train)
test_predictions = clf.predict(X_test)
print("测试集准确率:  %s " % accuracy_score(y_test, test_predictions))

predictions = clf.predict(test_pr)

print(predictions)
for i in range(0,len(predictions)):
    if y_result[i]==1 and predictions[i]==1 :
        TP= TP + 1
    if y_result[i]==0 and predictions[i]==0:
        TN = TN + 1
    if y_result[i]==0 and predictions[i]==1:
        FP=FP + 1
    if y_result[i]==1 and predictions[i]==0:
        FN=FN + 1        
print(TP ,TN ,FP ,FN)
        
print("最终准确率:  %s " % accuracy_score(y_result, predictions))
print('召回率：%s'%(TP/(TP+FN)))
print("精确率：%s"%(TP/(TP+FP)))
