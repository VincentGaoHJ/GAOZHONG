# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:13:25 2018

@author: gaoha
"""
import numpy as np
import csv

W2V = {}

W2V_file = csv.reader(open(".\data\gaozhong\gaozhong_english\W2V.csv","r",encoding = "utf-8"))

W2V_visualize = []

h = 0
for stu in W2V_file:
    w2v = stu[2].split(",")
# =============================================================================
#     for index, item in enumerate(w2v):
# #        print(item)
#         w2v[index] = float(item)
# =============================================================================
    w2v = "\t".join(w2v)
    W2V_visualize.append(w2v)
#    W2V_visualize[h] = w2v
#    if h % 1000 == 0 :
#        print(stu[1])
#        print(stu[2])
    h += 1
    
   
print(h)
#print(W2V_visualize.shape)
    

W2V = open('.\data\gaozhong\gaozhong_english\W2V_visualize.tsv','w',encoding = "utf_8_sig", newline='') # 设置newline，否则两行之间会空一行
writer = csv.writer(W2V)
# =============================================================================
# for i in range(h):
#     if i % 1000 == 1:
#         print(i)
# #    print(W2V_visualize[i])
#     writer.writerow(list(W2V_visualize[i]))
# W2V.close()
# =============================================================================


for i in range(len(W2V_visualize)):
#    print(W2V_visualize[i])
    writer.writerow(W2V_visualize[i])
W2V.close()