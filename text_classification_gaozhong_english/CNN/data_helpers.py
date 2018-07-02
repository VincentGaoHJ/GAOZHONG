import numpy as np
import csv
from collections import defaultdict

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


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    # 每次只输出shuffled_data[start_index:end_index]这么多

    data = np.array(data)
    print(data.shape)
    data_size = len(data)
    #每一个epoch有多少个batch_size
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            # 当前batch的索引开始
            start_index = batch_num * batch_size
            # 判断下一个batch是不是超过最后一个数据了
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]