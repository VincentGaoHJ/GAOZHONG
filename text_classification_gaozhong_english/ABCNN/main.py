"""
** deeplean-ai.com **
** dl-lab **
created by :: GauravBh1010tt
"""

import sys
sys.path.append("..\_deeplearn_utils")

import model_abcnn as model
import wiki_utils as wk
from dl_text.metrics import eval_metric
from dl_text.metrics import precision_recall
from dl_text import dl
import csv
import numpy as np
import data_helpers


glove_fname = 'glove.6B/glove.6B.50d.txt'

################### DEFINING MODEL AND PREDICTION FILE ###################

lrmodel = model.abcnn
model_name = 'abcnn'

################### DEFINING HYPERPARAMETERS ###################

wordVec_model = {}
W2V_file = csv.reader(open(".\data\gaozhong\gaozhong_english\W2V.csv","r",encoding = "utf-8"))


for stu in W2V_file:
    wordVec_model[stu[1]] = stu[2]
    

# =============================================================================
# print(embedding_matrix)
# =============================================================================

dimx = 250
dimy = 250
dimft = 44
batch_size = 64
vocab_size = 64778
embedding_dim = 50
nb_filter = 120
filter_length = (50,4)
depth = 1
nb_epoch = 10
shared = 0
opt_params = [0.001,'adam']
    
print("Loading data...")
x_text, x_text_r, y = data_helpers.load_data_and_labels(".\data\gaozhong\gaozhong_english\gaozhong_english_preprocessed.csv",".\data\gaozhong\gaozhong_english\labels_english.csv")
#print(x_text)
y = np.array(y)


# =============================================================================
# ques, ans, label_train, train_len, test_len,\
#          _, res_fname, pred_fname, feat_train, feat_test = wk.load_wiki(model_name, glove_fname)
# =============================================================================
     
         
# =============================================================================
# print(ques[0])
# print(x_text[0])
# =============================================================================
         
# =============================================================================
# data_l , data_r, embedding_matrix = dl.process_data(ques, ans,
#                                                  wordVec_model,dimx=dimx,
#                                                  dimy=dimy,vocab_size=vocab_size,
#                                                  embedding_dim=embedding_dim)
# =============================================================================

# =============================================================================
# data_l , data_r, embedding_matrix = dl.process_data(x_text, x_text_r,
#                                                  wordVec_model,dimx=dimx,
#                                                  dimy=dimy,vocab_size=vocab_size,
#                                                  embedding_dim=embedding_dim)
# =============================================================================



length = 250

a = np.zeros(shape=(len(y),length,50))


W2V = {}

W2V_file = csv.reader(open(".\data\gaozhong\gaozhong_english\W2V.csv","r",encoding = "utf-8"))

h = 1
for stu in W2V_file:
    wordVec_model[stu[1]] = stu[2]
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
        a[m][k] = wordVec_model[i].split(",")[:50]
        k += 1     
    m += 1
    
print("a.shape = {}".format(a.shape))
print("y.shape = {}".format(y.shape))

x_text = a
del a
x_text_r = x_text

# Randomly shuffle data
np.random.seed(10) # 使得随机数列可预测
# 产生一个array：起始点0，结束点len(y)，步长1。然后将其打乱。
shuffle_indices = np.random.permutation(np.arange(len(y))) 
x_shuffled = x_text[shuffle_indices] # 将文件句子和标签以同样的方式打乱
xr_shuffled = x_text_r[shuffle_indices] # 将文件句子和标签以同样的方式打乱
y_shuffled = y[shuffle_indices]


# Split train/test set
#直接把dev_sample_index前用于训练,dev_sample_index后的用于训练;
dev_sample_index = -1 * int(0.1 * float(len(y))) # -1:代表从后往前取
X_train_l, X_test_l = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
X_train_r, X_test_r = xr_shuffled[:dev_sample_index], xr_shuffled[dev_sample_index:]
label_train, label_test = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]


print(X_train_l)


if model_name == 'abcnn':
    lrmodel = lrmodel(dimx=dimx, dimy=dimy, nb_filter = nb_filter, embedding_dim = embedding_dim, 
                      filter_length = filter_length, depth = depth, shared = shared,
                      opt_params = opt_params)
    
    print ('\n',model_name,'model built \n')
    lrmodel.fit([X_train_l, X_train_r],label_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=2)
#    map_val, mrr_val = eval_metric(lrmodel, X_test_l, X_test_r, res_fname, pred_fname, feat_test=feat_test)
    pre, rec, pre_2, rec_2, pre_3, rec_3 = precision_recall(lrmodel, X_test_l, X_test_r, label_test)
    print(pre, rec, pre_2, rec_2, pre_3, rec_3)

else:
    lrmodel = lrmodel(embedding_matrix, dimx=dimx, dimy=dimy, nb_filter = nb_filter, embedding_dim = embedding_dim, 
                      filter_length = filter_length, depth = depth, shared = shared,
                      opt_params = opt_params)
    
    print ('\n', model_name,'model built \n')
    lrmodel.fit([X_train_l, X_train_r],label_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=2)
#    map_val, mrr_val = eval_metric(lrmodel, X_test_l, X_test_r, res_fname, pred_fname)
    map_val, mrr_val = eval_metric(lrmodel, X_test_l, X_test_r, res_fname, pred_fname)


print ('MAP : ',map_val,' MRR : ',mrr_val)
