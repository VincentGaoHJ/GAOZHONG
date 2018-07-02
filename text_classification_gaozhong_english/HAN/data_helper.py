#coding=utf-8
import os
import pickle
import nltk
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict
import csv

# 使用nltk分词分句器
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = WordPunctTokenizer()


def build_vocab(vocab_path, yelp_json_path, vocab_reverse_path):

    if os.path.exists(vocab_path):
        vocab_file = open(vocab_path, 'rb')
        vocab = pickle.load(vocab_file)
        print ("load vocab finish!")
    else:
        # 记录每个单词及其出现的频率
        word_freq = defaultdict(int)
        # 读取数据集，并进行分词，统计每个单词出现次数，保存在word freq中
        
        with open(yelp_json_path, encoding = "utf-8") as csvFile:
            readCSV = csv.reader(csvFile, delimiter = ",")
            for row in readCSV:
                row = "".join(row)
                words = word_tokenizer.tokenize(row)
#                print(words)
                for word in words:
                    word_freq[word] += 1
            print ("load finished")

        # 构建vocablary，并将出现次数小于5的单词全部去除，视为UNKNOW
        vocab = {}
        i = 1
        vocab['UNKNOW_TOKEN'] = 0
        for word, freq in word_freq.items():
            if freq > 0:
                vocab[word] = i
                i += 1

        vocab_reverse = {v : k for k, v in vocab.items()}
        
        # 将逆向词汇表保存下来
        with open(vocab_reverse_path, 'wb') as g:
            pickle.dump(vocab_reverse, g)
            print ("vocab_reverse save finished")
            
        # 将词汇表保存下来
        with open(vocab_path, 'wb') as g:
            pickle.dump(vocab, g)
            print (len(vocab))  # 159654
            print ("vocab save finished")

    return vocab, vocab_reverse

def load_dataset(yelp_json_path, labels_json_path, max_sent_in_doc, max_word_in_sent):
    yelp_data_path = yelp_json_path[0:-5] + "_data.pickle"
    vocab_path = yelp_json_path[0:-5] + "_vocab.pickle"
    vocab_reverse_path = yelp_json_path[0:-5] + "_vocab_reverse.pickle"
    doc_num = 17606 #数据个数
    if not os.path.exists(yelp_data_path):

        vocab, vocab_reverse = build_vocab(vocab_path, yelp_json_path, vocab_reverse_path)
        num_classes = 236
        UNKNOWN = 0

        data_x = np.zeros([doc_num,max_sent_in_doc,max_word_in_sent])
        data_y = []

        #将所有的评论文件都转化为30*30的索引矩阵，也就是每篇都有30个句子，每个句子有30个单词
        # 不够的补零，多余的删除，并保存到最终的数据集文件之中
        
        
        with open(yelp_json_path, encoding = "utf-8") as csvFile:
            readCSV = csv.reader(csvFile, delimiter = ",")
            k = 0
            for row in readCSV:
                row = "".join(row)
                sents = sent_tokenizer.tokenize(row)
                doc = np.zeros([max_sent_in_doc, max_word_in_sent])
                for i, sent in enumerate(sents):
                    if i < max_sent_in_doc:
                        word_to_index = np.zeros([max_word_in_sent],dtype=int)
                        for j, word in enumerate(word_tokenizer.tokenize(sent)):
                            if j < max_word_in_sent:
                                    word_to_index[j] = vocab.get(word, UNKNOWN)
                        doc[i] = word_to_index

                data_x[k] = doc
                k += 1
            
            
            
        with open(labels_json_path, encoding = "utf-8") as csvFile2:
            readCSV = csv.reader(csvFile2, delimiter = ",")
            for row in readCSV:
                d = defaultdict(list)
                for k,va in [(v,i) for i,v in enumerate(row)]:
                    d[k].append(va)
                labels = [0] * num_classes
                for k in range(len(d.get("1.0"))):
                    index = d.get("1.0")[k]
                    labels[index] = 1
                for k in range(len(d.get("0.0"))):
                    index = d.get("0.0")[k]
                    labels[index] = 0
                
#                print(len(row))
                data_y.append(labels)          
                
        # Randomly shuffle data
        np.random.seed(10) # 使得随机数列可预测
        # 产生一个array：起始点0，结束点len(y)，步长1。然后将其打乱。
        shuffle_indices = np.random.permutation(np.arange(len(data_y))) 
        x_shuffled = data_x[shuffle_indices] # 将文件句子和标签以同样的方式打乱
        y_shuffled = np.array(data_y)[shuffle_indices]
                
        pickle.dump((x_shuffled, y_shuffled), open(yelp_data_path, 'wb'))
        print (len(x_shuffled)) #229907


    else:
        data_file = open(yelp_data_path, 'rb')
        x_shuffled, y_shuffled = pickle.load(data_file)
        
        vocab_reverse_file = open(vocab_reverse_path, 'rb')
        vocab_reverse = pickle.load(vocab_reverse_file)


    
    length = len(x_shuffled)
    train_x, dev_x = x_shuffled[:int(length*0.9)], x_shuffled[int(length*0.9)+1 :]
    train_y, dev_y = y_shuffled[:int(length*0.9)], y_shuffled[int(length*0.9)+1 :]
    

    return train_x, train_y, dev_x, dev_y, vocab_reverse

if __name__ == '__main__':
    load_dataset("data/gaozhong_english/gaozhong_english_preprocessed.csv", "data/gaozhong_english/labels_english.csv", 10, 10)
#    load_dataset("data/yelp_academic_dataset_review.json", "data/yelp_academic_dataset_review.json", 30, 30)

