# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:10:29 2018

@author: gaoha
"""

import csv
import re

data_source = 'data/gaozhong/gaozhong_english/gaozhong_english_preprocessed.csv'

alphabet_raw = {}
alphabet = {}
alphabet_str = []

with open(data_source, encoding = "utf-8") as csvFile:
    readCSV = csv.reader(csvFile, delimiter = ",")
    for row in readCSV:
        row = "".join(row)
        row = row.lower()
        for x in row:
            if x in alphabet_raw:
                alphabet_raw[x] += 1
            else:
                alphabet_raw[x] = 1
                
#print(alphabet)
for key,value in alphabet_raw.items():
    if value > 100:
        alphabet[key] = value
                
print(alphabet)
alphabet = sorted(alphabet.items(),key = lambda x:x[1],reverse = True)
pattern = re.compile("'(.*)'")
for i in range(len(alphabet)):
    x = pattern.findall(str(alphabet[i]))
    x = "".join(x)
    alphabet_str.append(x)
alphabet_str = "".join(alphabet_str)
print(alphabet_str)
    
print(len(alphabet))