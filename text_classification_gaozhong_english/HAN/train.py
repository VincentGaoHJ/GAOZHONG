#coding=utf-8
import tensorflow as tf
import time
import os
from data_helper import load_dataset
from HAN_model import HAN
import numpy as np
import csv


# Data loading params
tf.flags.DEFINE_string("yelp_json_path", "data/gaozhong_english/gaozhong_english_preprocessed.csv", "data directory")
tf.flags.DEFINE_string("labels_json_path", "data/gaozhong_english/labels_english.csv", "data directory")
tf.flags.DEFINE_integer("vocab_size", 63585, "vocabulary size")
tf.flags.DEFINE_integer("num_classes", 236, "number of classes")
tf.flags.DEFINE_integer("embedding_size", 128, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_sent_in_doc", 30, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_word_in_sent", 30, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 300, "evaluate every this many batches")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")


FLAGS = tf.flags.FLAGS

print(FLAGS.max_sent_in_doc)
print(FLAGS.max_word_in_sent)

train_x, train_y, dev_x, dev_y, vocab = load_dataset(FLAGS.yelp_json_path, FLAGS.labels_json_path, FLAGS.max_sent_in_doc, FLAGS.max_word_in_sent)
print ("data load finished")

#print(train_x)

with tf.Session() as sess:
    han = HAN(vocab_size=FLAGS.vocab_size,
                    num_classes=FLAGS.num_classes,
                    embedding_size=FLAGS.embedding_size,
                    hidden_size=FLAGS.hidden_size)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=han.input_y,
                                                                      logits=han.out,
                                                                      name='loss'))
    with tf.name_scope('accuracy'):
        predict = tf.argmax(han.out, axis=1, name='predict')
        label = tf.argmax(han.input_y, axis=1, name='label')
        acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))
        
        
        

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.grad_clip)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            grad_summaries.append(grad_hist_summary)

    grad_summaries_merged = tf.summary.merge(grad_summaries)

    loss_summary = tf.summary.scalar('loss', loss)
    acc_summary = tf.summary.scalar('accuracy', acc)


    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.max_sentence_num: FLAGS.max_sent_in_doc,
            han.max_sentence_length: FLAGS.max_word_in_sent,
            han.batch_size: FLAGS.batch_size
        }
        _, step, summaries, cost, accuracy, out, doc_vec, doc_encoded, sent_vec, word_encoded, word_embedded, input_x, input_y = sess.run([train_op, global_step, train_summary_op, loss, acc, han.out, han.doc_vec, han.doc_encoded, han.sent_vec, han.word_encoded, han.word_embedded, han.input_x, han.input_y], feed_dict)

        time_str = str(int(time.time()))        
        scores = out
        
        
# =============================================================================
#         print(input_x.shape)
#         print(input_y.shape)
#         print(word_embedded.shape)
#         print(word_encoded.shape)
#         print(sent_vec.shape)
#         print(doc_encoded.shape)
#         print(doc_vec.shape)
#         print(scores.shape)
#         print(y_batch)
# =============================================================================
        
        y_indices = scores.argsort()[:, -1:][:, ::-1]
#        print(y_indices)
        pre = 0.0
        rec = 0.0
        for i in range(len(y_batch)):
            intersec_true = 0
            for j in y_indices[i]:
                intersec_true += y_batch[i][j]
            true_total_count = np.count_nonzero(y_batch[i])
            pred_total_count = len(y_indices[i])
            pre += intersec_true*1.0/pred_total_count
            rec += intersec_true*1.0/true_total_count
        pre = pre/len(y_batch)
        rec = rec/len(y_batch)
        
        
        
        y_indices_2 = scores.argsort()[:, -2:][:, ::-1]
#        print(y_indices_2)
        pre_2 = 0.0
        rec_2 = 0.0
        for i in range(len(y_batch)):
            intersec_true = 0
            for j in y_indices_2[i]:
                intersec_true += y_batch[i][j]
            true_total_count = np.count_nonzero(y_batch[i])
            pred_total_count = len(y_indices_2[i])
            pre_2 += intersec_true*1.0/pred_total_count
            rec_2 += intersec_true*1.0/true_total_count
        pre_2 = pre_2/len(y_batch)
        rec_2 = rec_2/len(y_batch)
        
        
        y_indices_3 = scores.argsort()[:, -3:][:, ::-1]
#        print(y_indices_3)
        pre_3 = 0.0
        rec_3 = 0.0
        for i in range(len(y_batch)):
            intersec_true = 0
            for j in y_indices_3[i]:
                intersec_true += y_batch[i][j]
            true_total_count = np.count_nonzero(y_batch[i])
            pred_total_count = len(y_indices_3[i])
            pre_3 += intersec_true*1.0/pred_total_count
            rec_3 += intersec_true*1.0/true_total_count
        pre_3 = pre_3/len(y_batch)
        rec_3 = rec_3/len(y_batch)
        
        
# =============================================================================
#         print('\n' * 2)                    
#         print("{}: step {}, loss {:g}".format(time_str, step, cost))
#         print("{}: pre_1 {}, rec_1 {:g}".format(time_str, pre, rec))
#         print("{}: pre_2 {}, rec_2 {:g}".format(time_str, pre_2, rec_2))
#         print("{}: pre_3 {}, rec_3 {:g}".format(time_str, pre_3, rec_3))
#         print('\n' * 2)        
# =============================================================================
        
#        print("{}: step {}, loss {:g}, pre {:g}, rec {:g}, pre_2 {:g}, rec_2 {:g}, pre_3 {:g}, rec_3 {:g}".format(time_str, step, cost, pre, rec, pre_2, rec_2, pre_3, rec_3))
        train_summary_writer.add_summary(summaries, step)

        return step

    def dev_step(x_batch, y_batch, writer=None):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.max_sentence_num: FLAGS.max_sent_in_doc,
            han.max_sentence_length: FLAGS.max_word_in_sent,
            han.batch_size: FLAGS.batch_size
        }
        step, summaries, cost, accuracy, out = sess.run([global_step, dev_summary_op, loss, acc, han.out], feed_dict)
        time_str = str(int(time.time()))
        
        scores = out
        
        y_indices = scores.argsort()[:, -1:][:, ::-1]
        pre = 0.0
        rec = 0.0
        for i in range(len(y_batch)):
            intersec_true = 0
            for j in y_indices[i]:
                intersec_true += y_batch[i][j]
            true_total_count = np.count_nonzero(y_batch[i])
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
            true_total_count = np.count_nonzero(y_batch[i])
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
            true_total_count = np.count_nonzero(y_batch[i])
            pred_total_count = len(y_indices_3[i])
            pre_3 += intersec_true*1.0/pred_total_count
            rec_3 += intersec_true*1.0/true_total_count
        pre_3 = pre_3/len(y_batch)
        rec_3 = rec_3/len(y_batch)
        
        
        print('\n' * 2)                    
        print("{}: step {}, loss {:g}".format(time_str, step, cost))
        print("{}: pre_1 {}, rec_1 {:g}".format(time_str, pre, rec))
        print("{}: pre_2 {}, rec_2 {:g}".format(time_str, pre_2, rec_2))
        print("{}: pre_3 {}, rec_3 {:g}".format(time_str, pre_3, rec_3))
        print('\n' * 2)        
        
        
        if writer:
            writer.add_summary(summaries, step)
            
        writeFile2 = open('.\\data\\gaozhong_english\\test_data_english_eval.csv','a+', newline='') # 设置newline，否则两行之间会空一行
        writer2 = csv.writer(writeFile2)
        writer2.writerow([step, cost, pre, rec, pre_2, rec_2, pre_3, rec_3])
        writeFile2.close()

    for epoch in range(FLAGS.num_epochs):
        print('\n' * 2)
        print('current epoch %s' % (epoch + 1))
        for i in range(0, 15800, FLAGS.batch_size):
            x = train_x[i:i + FLAGS.batch_size]
            y = train_y[i:i + FLAGS.batch_size]
            step = train_step(x, y)
            if step % FLAGS.evaluate_every == 0:
                dev_step(dev_x, dev_y, dev_summary_writer)