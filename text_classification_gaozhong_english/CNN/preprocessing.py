# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:48:32 2018

@author: gaoha
"""

import re
import operator
import csv
import numpy as np
import jieba


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    #清理掉无词义的符号
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean(file):
    
    file = re.sub(r"\d{10}\t23\t.{1,20}\t\r\n", "\d{10}\t23\t.{1,20}\t", file)
    file = re.sub(r"\d{10}\t23\t","" , file)
    
    return file
    
    


def load_data(file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    file = open(file, "r", encoding='UTF-8')
    file = file.read()
    file = clean(file)
    
    file_list = re.split(r'[\r\n]',file)
    return file_list

def class_extraction(file_list):
    h = 0
    class_list = []
    class_dict = {}

    for i in range(len(file_list)):
        if i > len(file_list):
            break
        if file_list[i] == "" :
            del file_list[i]
            continue
        x = re.findall(r".{1,50}\t",file_list[i])
        x = str(x)
        x = re.sub("\\\\t","",x)
        x = re.sub(r"'","",x)
        x = list(x)
        x[-1] = ""
        x[0] = ""
        x = ''.join(x)
#        file_list[i] = re.sub(r".{1,50}\t",'' , file_list[i])
        if ";" in x :
            x = x.split(";")
            for k in range(len(x)):
                if x[k] != "" :
                    if x[k] not in class_list :
                        h += 1
                        class_list.append(x[k])
                        class_dict.update({x[k]:1})
                        
                    else:
                        h += 1
                        class_dict[x[k]] += 1


        else:
            if x != "" :
                if x not in class_list :
                    h += 1
                    class_list.append(x)
                    class_dict.update({x:1})
                else:
                    h += 1
                    class_dict[x] += 1

    return class_list, class_dict, h, file_list




def generate_labels(file_list, sorted_class_list):
    
    y = np.zeros(shape=(len(file_list), len(sorted_class_list)))
    for i in range(len(file_list)):
        if i > len(file_list):
            break
        x = re.findall(r".{1,50}\t",file_list[i])
        x = str(x)
        x = re.sub("\\\\t","",x)
        x = re.sub(r"'","",x)
        x = list(x)
        x[-1] = ""
        x[0] = ""
        x = ''.join(x)
#        print(x)
#        print(file_list[i])
        file_list[i] = re.sub(r".{1,50}\t",'' , file_list[i])
#        print(file_list[i])
        if ";" in x :
            x = x.split(";")
#            print(x)
            for k in range(len(x)):
#                print(x[k])
                
                index = sorted_class_list.index(x[k])
#                print("=====")
#                print(index)
                y[i, index] = 1

        else:
            index = sorted_class_list.index(x)
            y[i, index] = 1
#            print(index)            
    return y
    
                        
def fenci(x):
    for i in range(len(x)):
        x[i] = jieba.cut(x[i])
        x[i] = " ".join(x[i])
        x[i] = re.sub(r"\s{2,}", " ", x[i])
        x[i] = x[i].strip()
#        print(x[i])
    return x
            
    
     
filePath = ".\data\gaozhong\gaozhong_english\gaozhong_english.txt"
x = load_data(filePath)
class_list, class_dict, h, file_list = class_extraction(x)

#print(class_dict)
sorted_class_tuple=sorted(class_dict.items(), key = lambda item:item[1], reverse = True)
#print(sorted_class_tuple)

sorted_class_list = list(sorted_class_tuple)
for i in range(len(sorted_class_list)):
    result = re.findall(".*'(.*)'.*",str(sorted_class_list[i]))
#    print(result[0])
    sorted_class_list[i] = result[0]

#print(sorted_class_list)
print("=================")
print("=================")
print("类别数量为：")
print(len(class_list))
print(len(class_dict))


y = generate_labels(x , sorted_class_list)

print("=================")
print("=================")
print(y)
print("文件数量为：")
print(len(file_list))
print(y.shape)

file_list = fenci(file_list)

csvFile =  open(".\data\gaozhong\gaozhong_english\labels_english.csv", "w", encoding='utf-8', newline='')
writer = csv.writer(csvFile)
for i in range(len(y)):
    writer.writerow(y[i])
csvFile.close()

print("标签已写入文件")



csvFile2 =  open(".\data\gaozhong\gaozhong_english\sorted_class_list.csv", "w", newline='')
writer = csv.writer(csvFile2)
for i in range(len(sorted_class_list)):
    writer.writerow(sorted_class_list[i])
csvFile2.close()

print("类别名称已写入文件")


csvFile3 =  open(".\data\gaozhong\gaozhong_english\gaozhong_english_preprocessed.csv", "w", encoding='utf_8_sig', newline='')
writer = csv.writer(csvFile3)
for i in range(len(file_list)):
    writer.writerow([file_list[i]])
csvFile3.close()

print("预处理后数据已写入文件")


