# -*- coding: utf-8 -*-
"""
Created on Fri May  4 17:54:54 2018

@author: gaoha
"""

import numpy as np
import csv
from collections import defaultdict

x_text = []
y = []

data_file = ".\data\gaozhong\gaozhong_english\gaozhong_english_preprocessed.csv"
labels_file = ".\data\gaozhong\gaozhong_english\labels_english.csv"

with open(data_file, encoding = "utf-8") as csvFile:
    readCSV = csv.reader(csvFile, delimiter = ",")
    for row in readCSV:
        row = "".join(row)
        x_text.append(row)


with open(labels_file, encoding = "utf-8") as csvFile2:
    readCSV = csv.reader(csvFile2, delimiter = ",")
    for row in readCSV:
        y.append(row)
        

        
print("x = {}".format(len(x_text)))
print("y = {}".format(len(y)))


csvFile3 = open(".\\data\\gaozhong\\gaozhong_english\\file_length.csv",'w', newline='') # 设置newline，否则两行之间会空一行
writer = csv.writer(csvFile3)
for i in range(len(y)):
    length = len(x_text[i].split(" "))
    d = defaultdict(list)
    for k,va in [(v,i) for i,v in enumerate(y[i])]:
        d[k].append(va)

    for k in range(len(d.get("1.0"))):
        index = d.get("1.0")[k]
        writer.writerow([index, length])
csvFile3.close()