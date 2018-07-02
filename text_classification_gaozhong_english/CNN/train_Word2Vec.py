# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 13:44:32 2018

@author: gaoha
"""


from sklearn.manifold import TSNE
from Word2Vec import Word2Vec
import csv


# =============================================================================
# # 第一步：准备数据，这里首先将网络上的资源下载下来
# url = 'http://mattmahoney.net/dc/'
# filename = maybe_download('text8.zip', 31344016)  # do not change this line
# # Now we have the data
# words = Word2Vec.read_data(filename)
# print('Data size', len(words))  # The length of "words", not bytes
# =============================================================================

def step_1(filename) :
    # 第一步：读取本地数据
    # Now we have the data
    
    words = Word2Vec.read_data(filename)
    #print(words)
    print('Data size', len(words))  # The length of "words", not bytes
    print("第一步已经完成：读取本地数据")
    
    return words

def step_2(words) :
    # 第二步：建立一个字典并且将不常用的字符用UNK替换掉（这里只用50000个词建立字典）
    vocabulary_size = 65000
    data, count, dictionary, reverse_dictionary = Word2Vec.build_dataset(words, vocabulary_size)
    del words  # 删除列表中某一个元素或者是某一个片段,只需要给出元素所在的索引值,而不需要给出元素的具体值。
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
    data_index = 0
    print("第二步已经完成：建立一个字典并且将不常用的字符用UNK替换掉")
    return data, count, dictionary, reverse_dictionary, data_index, vocabulary_size

def step_3(data, data_index, reverse_dictionary) :
    
    # 第三步：为Skip-Gram模型生成一个训练集
    batch, labels = Word2Vec.generate_batch(
            data, data_index, batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->',
              labels[i, 0], reverse_dictionary[labels[i, 0]])
    print("第三步已经完成：为Skip-Gram模型生成一个训练集")
    return batch, labels

def step_4(vocabulary_size):        
    # 第四步：建立并且训练一个skip-gram model.
    init, similarity, optimizer, graph, batch_size, num_skips, batch_size, num_skips, skip_window, train_inputs, train_labels, loss,normalized_embeddings = Word2Vec.establishModel(vocabulary_size)
    print("第四步已经完成：建立并且训练一个skip-gram model")
    return init, similarity, optimizer, graph, batch_size, num_skips, batch_size, num_skips, skip_window, train_inputs, train_labels, loss,normalized_embeddings

def step_5(init, similarity, optimizer, graph, batch_size, num_skips, skip_window,
            train_inputs, train_labels, loss, data, data_index,
            normalized_embeddings, reverse_dictionary) :
    # 第五步：Begin training.
    final_embeddings = Word2Vec.training(
            init, similarity, optimizer, graph, batch_size, num_skips, skip_window,
            train_inputs, train_labels, loss, data, data_index,
            normalized_embeddings, reverse_dictionary)
    print(final_embeddings)
    print(final_embeddings.shape)
    print(len(reverse_dictionary))
    print("第五步已经完成：Begin training")    
    return final_embeddings

def step_6(final_embeddings,reverse_dictionary) :    
    #第六步：降维可视化处理    
    num_points = 400
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
    
    words = [reverse_dictionary[i] for i in range(1, num_points+1)]
    Word2Vec.plot(two_d_embeddings, words)
    
    print("已经全部完成：降维可视化处理")
    
def writeFile(final_embeddings, reverse_dictionary):
    W2V = open('.\data\gaozhong\gaozhong_english\W2V.csv','w',encoding = "utf_8_sig", newline='') # 设置newline，否则两行之间会空一行
    writer = csv.writer(W2V)
    i = 0
    for key in reverse_dictionary:
        if i % 1000 == 1:
            print(i)
# =============================================================================
#         for k in range(0, len(final_embeddings[i])):
#             final_embeddings[i][k] = str(final_embeddings[i][k])
# =============================================================================
        writer.writerow([key, reverse_dictionary[key], ",".join(map(str, final_embeddings[i]))])
        i += 1
    W2V.close()
    
    
def train():
    filename = ".\data\gaozhong\gaozhong_english\gaozhong_english_preprocessed.csv"    
    words = step_1(filename)
    data, count, dictionary, reverse_dictionary, data_index, vocabulary_size = step_2(words)
    batch, labels = step_3(data, data_index,reverse_dictionary)
    init, similarity, optimizer, graph, batch_size, num_skips, batch_size, num_skips, skip_window, train_inputs, train_labels, loss,normalized_embeddings = step_4(vocabulary_size)
    final_embeddings = step_5(init, similarity, optimizer, graph, batch_size, num_skips, skip_window,
            train_inputs, train_labels, loss, data, data_index,
            normalized_embeddings, reverse_dictionary)
    step_6(final_embeddings, reverse_dictionary)
    writeFile(final_embeddings, reverse_dictionary)
    
    
    
train()