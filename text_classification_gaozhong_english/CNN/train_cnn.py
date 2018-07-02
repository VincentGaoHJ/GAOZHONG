#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import re
import csv

# Parameters

# Data loading params
#语料文件路径定义
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", ".\data\gaozhong\gaozhong_english\gaozhong_english_preprocessed.csv", "Data source for the positive data.")
tf.flags.DEFINE_string("labels_file", ".\data\gaozhong\gaozhong_english\labels_english.csv", "Data source for the positive data.")

# Model Hyperparameters
#定义神经网络的超参数
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5,6", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
#训练参数
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 600, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 3000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


# 打印一下相关初始参数
FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
print(FLAGS.batch_size)
print(FLAGS.__flags)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.data_file, FLAGS.labels_file)
#print(x_text)
y = np.array(y)
print("y.shape = {}".format(y.shape))

#print(x_text)
# Build vocabulary
#计算最长句子的长度
max_document_length = max([len(x.split(" ")) for x in x_text])
print(max_document_length)
#文本的长度大于最大长度，那么它会被剪切，反之则用0填充。
#vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#fit_transform()的作用就是先拟合数据，然后转化它将其转化为标准形式
#x = np.array(list(vocab_processor.fit_transform(x_text)))

length = 250

a = np.zeros(shape=(len(y),length,128))


W2V = {}

W2V_file = csv.reader(open(".\data\gaozhong\gaozhong_english\W2V.csv","r",encoding = "utf-8"))

h = 1
for stu in W2V_file:
    W2V[stu[1]] = stu[2]
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
        a[m][k] = W2V[i].split(",")
        k += 1     
    m += 1
    
print("a.shape = {}".format(a.shape))
print("y.shape = {}".format(y.shape))

# Randomly shuffle data
np.random.seed(10) # 使得随机数列可预测
# 产生一个array：起始点0，结束点len(y)，步长1。然后将其打乱。
shuffle_indices = np.random.permutation(np.arange(len(y))) 
x_shuffled = a[shuffle_indices] # 将文件句子和标签以同样的方式打乱
y_shuffled = y[shuffle_indices]


# Split train/test set
#直接把dev_sample_index前用于训练,dev_sample_index后的用于训练;
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y))) # -1:代表从后往前取
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

print("max_document_length: {:d}".format(max_document_length))


del x, x_shuffled, y_shuffled

print("Vocabulary Size: {:d}".format(len(y)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
with tf.Graph().as_default():
    #记录设备指派情况：tf.ConfigProto(log_device_placement=True)
    #自动选择运行设备：tf.ConfigProto(allow_soft_placement=True)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        #导入卷积池化网络
        cnn = TextCNN(
            # 句子的长度（将所有句子填充之后）
            sequence_length=x_train.shape[1], 
            # 输出层的类数，本程序中为两个（正和负）
            num_classes=y_train.shape[1], 
            # 词汇的大小，这里需要定义嵌入层的大小
            vocab_size=len(y), 
            # 我们嵌入的维度:128。
            embedding_size=FLAGS.embedding_dim, 
            # 我们希望卷积滤波器覆盖的字数:3,4,5。
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))), 
            # 每个过滤器尺寸的过滤器数量。
            num_filters=FLAGS.num_filters, 
            # l2正则化项; l2_reg_lambda 中文'拉姆那',正则化参数,这里用0
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        # 为全局步骤(global step)计数; 目标函数小于时optimizer,训练结束
        global_step = tf.Variable(0, name="global_step", trainable=False)

        learning_rate = tf.train.exponential_decay(0.001,
                                                   global_step = global_step,
                                                   decay_steps = 500,
                                                   decay_rate = 0.99)
        
        optimizer = tf.train.AdamOptimizer(learning_rate) # 定义优化器
        
        # 导入cnn.loss求偏导的差(也就是结果的变化量),反过来就知道,上次的变化,对结果影响
        # 从而知道是否到全局最优解
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)



        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        # 每一步都存参数,tensorboard可以看
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time())) # 文件名为时间
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        # 目标函数 和 准确率的参数保存
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        sco_summary = tf.summary.scalar("scores", cnn.scores)

        # Train Summaries
        # 训练数据保存
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        # 测试数据保存
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory
        # already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir) 
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
#        W2V.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            训练每一步,填入数据
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, scores, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op,cnn.scores, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            
            
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
            
            
                        
#            print("{}: step {}, loss {:g}, pre {:g}, rec {:g}, pre_2 {:g}, rec_2 {:g}, pre_3 {:g}, rec_3 {:g}".format(time_str, step, loss, pre, rec, pre_2, rec_2, pre_3, rec_3))

            train_summary_writer.add_summary(summaries, step)
            
            
            
            return step, loss, pre, rec, pre_2, rec_2, pre_3, rec_3

        def dev_step(x_batch, y_batch, writer=None):
            """
            测试每一步,填入数据
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0 # 神经元全部保留
            }
            step, summaries, scores, loss, accuracy, h, embedded_chars_expanded, pooled_outputs, h_pool, h_pool_flat, conv = sess.run(
                [global_step, dev_summary_op, cnn.scores, cnn.loss, cnn.accuracy,cnn.h, cnn.embedded_chars_expanded, cnn.pooled_outputs, cnn.h_pool, cnn.h_pool_flat, cnn.conv],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            
# =============================================================================
#             
#             print(embedded_chars_expanded.shape) 
#             print(h.shape)
#             print(conv.shape)
#             print(pooled_outputs) 
#             print(h_pool.shape) 
#             print(h_pool_flat)      
# =============================================================================
            
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
            
            
                        
            print("{}: step {}, loss {:g}".format(time_str, step, loss))
            print("{}: pre_1 {}, rec_1 {:g}".format(time_str, pre, rec))
            print("{}: pre_2 {}, rec_2 {:g}".format(time_str, pre_2, rec_2))
            print("{}: pre_3 {}, rec_3 {:g}".format(time_str, pre_3, rec_3))
            
            if writer:
                writer.add_summary(summaries, step)
                
            return step, loss, pre, rec, pre_2, rec_2, pre_3, rec_3

        # Generate batches
        print("==================")
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        print(batches)
        print("==================")

           
        
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch) # 按batch把数据拿进来
            step, loss, pre, rec, pre_2, rec_2, pre_3, rec_3 = train_step(x_batch, y_batch)
            # 将Session和global_step值传进来
            current_step = tf.train.global_step(sess, global_step) 
            if current_step % FLAGS.evaluate_every == 0:
                writeFile = open('.\\data\\gaozhong\\gaozhong_english\\test_data_english_train.csv','a+', newline='') # 设置newline，否则两行之间会空一行
                writer = csv.writer(writeFile)
                writer.writerow([step, loss, pre, rec, pre_2, rec_2, pre_3, rec_3])
                writeFile.close()
            # 每FLAGS.evaluate_every次每100执行一次测试
            if current_step % FLAGS.evaluate_every == 0: 
                print("\nEvaluation:")
                step, loss, pre, rec, pre_2, rec_2, pre_3, rec_3 = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                writeFile2 = open('.\\data\\gaozhong\\gaozhong_english\\test_data_english_eval.csv','a+', newline='') # 设置newline，否则两行之间会空一行
                writer2 = csv.writer(writeFile2)
                writer2.writerow([step, loss, pre, rec, pre_2, rec_2, pre_3, rec_3])
                writeFile2.close()
                print("")
            # 每checkpoint_every次执行一次保存模型
            if current_step % FLAGS.checkpoint_every == 0:
                # 定义模型保存路径
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
