# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:49:26 2018

@author: gaoha
"""

# First let's import all the necessary dependencies
import collections # 用于提供一些额外的数据类型
import math
import os # 提供了使用各种操作系统功能的接口
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from matplotlib import pylab
import csv

class Word2Vec:
    def __init__(self, graph, num_skips, batch_size, train_inputs,
                 skip_window, train_labels, loss, optimizer, final_embeddings):
        self.num_skips = num_skips
        self.batch_size = batch_size
        self.train_inputs = train_inputs
        self.skip_window = skip_window
        self.train_labels = train_labels
        self.loss = loss
        self.optimizer = optimizer
        self.train_inputs = train_inputs
        self.final_embeddings = final_embeddings
    
    # Read the data into a list of strings.
    def read_data(filename):
        """
        Extract the first file enclosed in a zip file as a list of words
        """

        with open(filename, encoding = 'utf_8_sig') as csvFile:
            readCSV = csv.reader(csvFile, delimiter = ",")
            x_text = []

            for row in readCSV:
                row = "".join(row)
                x_text.append(row)
                
            x_text_process = []
            
            for x in x_text:
                if x != "":
                    x_text_process.append(x)
        x_text_process = " ".join(x_text_process)
        data = tf.compat.as_str(x_text_process).split()

        return data
            

    
    
    def build_dataset(words, vocabulary_size):
    
        count = [['UNKo', -1]]
        
        #表示：使用collections.Counter统计words列表中单词的频数，然后使用most_common方法取频数为top（字典大小-1）的单词。然后加入到count中。
        count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
        
        #创建一个字典,将全部单词转为编号（以频数排序的编号），top50000之外的单词，认为UnKown,编号为0,并统计这类词汇的数量
        dictionary = dict()
        
        for word, _ in count:
            dictionary[word] = len(dictionary)
            
        data = list()
        unk_count = 0
        
        #遍历单词列表，
        for word in words:
            #对于其中每一个单词，先判断是否出现在dictionary中
            if word in dictionary:
                #如果出现，则转为其编号
                index = dictionary[word]
            else:
                #如果不是，则转为编号0
                index = 0  # dictionary['UNK']
                unk_count += 1
            # 将元素添加到已有list的末尾
            data.append(index)
        
        count[0][1] = unk_count
#        print(count)
        
        # 将dictionary的index和内容互换，也就是index为出现次数排序
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        print(reverse_dictionary)
        return data, count, dictionary, reverse_dictionary
    
    
    
    def generate_batch(data, data_index, batch_size, num_skips, skip_window):
        
        # 声明布尔值必须为真的判定，其返回值为假，就会触发异常。
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = (data_index + len(data) - span) % len(data)
        return batch, labels
    
    
    def establishModel(vocabulary_size):
        batch_size = 128
        embedding_size = 128  # Dimension of the embedding vector.
        skip_window = 1       # How many words to consider left and right.
        num_skips = 2         # How many times to reuse an input to generate a label.
        
        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        valid_size = 16     # Random set of words to evaluate similarity on.
        valid_window = 100  # Only pick dev samples in the head of the distribution.
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        num_sampled = 64    # Number of negative examples to sample.
        
        
        graph = tf.Graph()
        with graph.as_default():
        
            # Input data.
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        
            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                embeddings = tf.Variable(
                    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        
                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(
                    tf.truncated_normal([vocabulary_size, embedding_size],
                                        stddev=1.0 / math.sqrt(embedding_size)))
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=train_labels,
                               inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=vocabulary_size))
        
            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        
            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        
            # Add variable initializer.
            init = tf.global_variables_initializer()
            
            return init, similarity, optimizer, graph, batch_size, num_skips, batch_size, num_skips, skip_window, train_inputs, train_labels, loss, normalized_embeddings
    
    
    def training(init, similarity, optimizer, graph, batch_size, num_skips,
                 skip_window, train_inputs, train_labels, loss, data,
                 data_index, normalized_embeddings, reverse_dictionary):
        num_steps = 200001
        LOG_DIR = './log/'
        
        with tf.Session(graph = graph) as sess:
            # We must initialize all variables before we use them.
            init.run()
            print("Initialized")
        
            average_loss = 0
            for step in range(num_steps):
                batch_inputs, batch_labels = Word2Vec.generate_batch(
                        data, data_index, batch_size, num_skips, skip_window)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        
                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for sess.run()
                _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val
        
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0
                
            """
            Use TensorBoard to visualize our model. 
            This is not included in the TensorFlow website tutorial.
            """
# =============================================================================
#             words_to_visualize = 3000
#             final_embeddings = normalized_embeddings.eval(session=sess)[:words_to_visualize]
# =============================================================================
            final_embeddings = normalized_embeddings.eval(session=sess)
            embedding_var = tf.Variable(final_embeddings)
            sess.run(embedding_var.initializer)
            saver = tf.train.Saver([embedding_var])
            saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 0)
        
            # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
            config = projector.ProjectorConfig()
        
            # You can add multiple embeddings. Here we add only one.
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
        
            # Use the same LOG_DIR where you stored your checkpoint.
            summary_writer = tf.summary.FileWriter(LOG_DIR)
            summary_writer.add_graph(graph)
        
            # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
            # read this file during startup.
            projector.visualize_embeddings(summary_writer, config)
        
            # Write the metadata file to disk so that TensorBoard can find it later
#            labels = [(reverse_dictionary[i], i) for i in range(words_to_visualize)]
 #           DataFrame(labels, columns=['word', 'freq_rank']).to_csv('log/metadata.tsv', index=False, sep='\t')
                
        return final_embeddings
    
    def plot(embeddings, labels):
        assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
        pylab.figure(figsize=(15,15))  # in inches
        for i, label in enumerate(labels):
            x, y = embeddings[i,:]
            pylab.scatter(x, y)
            pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                           ha='right', va='bottom')
        pylab.show()





