# coding=utf-8
import tensorflow as tf
from data_helper import Dataset
import time
import os
from tensorflow.python import debug as tf_debug
from charCNN import CharCNN
import datetime
from config import config
import numpy as np
import csv

# Load data
print("正在载入数据...")
# 函数dataset_read：输入文件名,返回训练集,测试集标签
# 注：embedding_w大小为vocabulary_size × embedding_size
train_data = Dataset(config.data_source, config.label_source)

train_data.dataset_read()


print ("得到15845维的doc_train，label_train")
print ("得到1760维的doc_dev, label_train")


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    with sess.as_default():
        cnn = CharCNN(
            l0=config.l0,
            num_classes=config.nums_classes,
            conv_layers=config.model.conv_layers,
            fc_layers=config.model.fc_layers,
            l2_reg_lambda=0)
        # cnn = CharConvNet()
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(config.model.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
#        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def precision_recall(scores, y_batch):        
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
            
            return pre, rec, pre_2, rec_2, pre_3, rec_3


        def dev_batch_iter(data):
            """
            Generates a batch iterator for a dataset.
            """
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int((len(data)-1)/64) + 1
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * 64
                end_index = min((batch_num + 1) * 64, data_size)
                yield data[start_index:end_index]



        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: config.model.dropout_keep_prob
            }
            _, step, summaries, loss, y_pred = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.y_pred],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            
            
#            pre, rec, pre_2, rec_2, pre_3, rec_3 = precision_recall(y_pred, y_batch)
            
            print("{}: step {}, loss {:g}".format(time_str, step, loss))
# =============================================================================
#             print("{}: pre_1 {}, rec_1 {:g}".format(time_str, pre, rec))
#             print("{}: pre_2 {}, rec_2 {:g}".format(time_str, pre_2, rec_2))
#             print("{}: pre_3 {}, rec_3 {:g}".format(time_str, pre_3, rec_3))
# =============================================================================
            
            
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, y_pred = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.y_pred],
                feed_dict)

            pre, rec, pre_2, rec_2, pre_3, rec_3 = precision_recall(y_pred, y_batch)
            
            if writer:
                writer.add_summary(summaries, step)
            
            return pre, rec, pre_2, rec_2, pre_3, rec_3, loss
            
            
            
            
            

            
            



        print ("初始化完毕，开始训练")
        for i in range(config.training.epoches):
            batch_train = train_data.next_batch()
            # 训练模型
            train_step(batch_train[0], batch_train[1])
            current_step = tf.train.global_step(sess, global_step)
            # train_step.run(feed_dict={x: batch_train[0], y_actual: batch_train[1], keep_prob: 0.5})
            # 对结果进行记录
            if current_step % config.training.evaluate_every == 0:
                batch_train = train_data.dev_batch()
                print("\nEvaluation:")
                print("")
                i = 0
                sum_loss = 0
                sum_pre = 0
                sum_rec = 0
                sum_pre_2 = 0
                sum_rec_2 = 0
                sum_pre_3 = 0
                sum_rec_3 = 0       
                test_batches = dev_batch_iter(list(zip(batch_train[0], batch_train[1])))
                for test_batch in test_batches:
                    x_test_batch, y_test_batch = zip(*test_batch)
                    pre, rec, pre_2, rec_2, pre_3, rec_3, test_loss = dev_step(x_test_batch, y_test_batch)
                    sum_loss += test_loss
                    sum_pre += pre
                    sum_rec += rec
                    sum_pre_2 += pre_2
                    sum_rec_2 += rec_2
                    sum_pre_3 += pre_3
                    sum_rec_3 += rec_3   
                    i += 1
                    print(i)
                sum_pre = sum_pre/i
                sum_rec = sum_rec/i
                sum_pre_2 = sum_pre_2/i
                sum_rec_2 = sum_rec_2/i
                sum_pre_3 = sum_pre_3/i
                sum_rec_3 = sum_rec_3/i

                time_str = datetime.datetime.now().isoformat()
                print("{}: Evaluation Summary, Loss {:g}".format(time_str, sum_loss/i))
                print("pre_1 {}, rec_1 {:g}".format(sum_pre, sum_rec))
                print("pre_2 {}, rec_2 {:g}".format(sum_pre_2, sum_rec_2))
                print("pre_3 {}, rec_3 {:g}".format(sum_pre_3, sum_rec_3))  



                writeFile = open(".\\data\\gaozhong\\gaozhong_english\\test_data_english_eval.csv",'a+',newline="")
                writer = csv.writer(writeFile)
                writer.writerow([current_step, sum_loss/i, sum_pre, sum_rec, sum_pre_2, sum_rec_2, sum_pre_3, sum_rec_3])
                writeFile.close()                
                    
                
                
                
                
            if current_step % config.training.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
