import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    进行超参数配置
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        """
        参数设置：
        sequence_length：句子的长度（将所有句子填充之后）
        num_classes：输出层的类数，本程序中为两个（正和负）
        vocab_size：词汇的大小，这里需要定义嵌入层的大小
        embedding_size：我们嵌入的维度。
        filter_sizes：我们希望卷积滤波器覆盖的字数。我们将num_filters在这里指定每种尺寸。
                      例如，[3, 4, 5]意味着我们将有过滤器分别滑过3个，4个和5个词，总共3 * num_filters过滤器。
        num_filters：每个过滤器尺寸的过滤器数量。
        """

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0) # 先不用，写0

        # Embedding layer 
        """
        嵌入层
        作用：将词汇编成索引映射到低维向量中表示。它本质上是一个查找表（我们从数据中学习到的）
             通过tf.name_scope指定"embedding"
        """
        #创建一个命名空间，为了更好地在tensorboard中可视化
        with tf.device('/cpu:0'), tf.name_scope("embedding"): 
            # 用随机均匀分布来初始化嵌入层的矩阵
            # 定义W并初始化
# =============================================================================
#             self.W = tf.Variable(
#                 tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
#                 name="W")
#             # tf.nn.embedding_lookup：根据train_inputs中的id，寻找embeddings中的对应元素。
#             #形成一个三维张量：[None, sequence_length, embedding_size]
#             self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
# =============================================================================
            
            self.embedded_chars = self.input_x
            #手动添加一个维度：通道维度
            #形成一个四维张量：[None, sequence_length, embedding_size, 1]，以满足卷积的操作
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        """
        卷积层和池化层
        """
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                # 4个参数分别为filter_size高h，embedding_size宽w，channel为1，filter个数
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # W进行高斯初始化
                # stddev:标准差
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # b给初始化为一个常量
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    # 指定每一维的步长
                    strides=[1, 1, 1, 1],
                    # 这里不需要padding
                    padding="VALID",
                    name="conv")
                #输出的shape仍然是[batch, height, width, channels]

                # Apply nonlinearity 激活函数
                # 可以理解为,正面或者负面评价有一些标志词汇,这些词汇概率被增强，
                # 即一旦出现这些词汇,倾向性分类进正或负面评价,该激励函数可加快学习进度，
                # 增加稀疏性,因为让确定的事情更确定,噪声的影响就降到了最低。
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                
                self.conv = conv
                self.h = h
                
                # Maxpooling over the outputs
                # 池化
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1], 
                    # (h-filter+2padding)/strides+1=h-f+1
                    strides=[1, 1, 1, 1],
                    padding='VALID', # 这里不需要padding
                    name="pool")
                pooled_outputs.append(pooled)
                
        self.pooled_outputs = pooled_outputs

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) # 3 * 128
        # 扁平化数据，跟全连接层相连
        self.h_pool = tf.concat(pooled_outputs, 3)
        # 拉平
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        # drop层,防止过拟合,参数为dropout_keep_prob
        # 过拟合的本质是采样失真,噪声权重影响了判断，如果采样足够多,足够充分,
        # 噪声的影响可以被量化到趋近事实,也就无从过拟合。
        # 即数据越大,drop和正则化就越不需要。
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)


        # Final (unnormalized) scores and predictions
        # 输出层
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes], #前面连扁平化后的池化操作
                initializer=tf.contrib.layers.xavier_initializer()) # 定义初始化方式
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # 损失函数导入
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # xw+b
            # 得分函数
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

            # 预测结果
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            # loss，交叉熵损失函数
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores,
                                                             labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss


        # Accuracy
        with tf.name_scope("accuracy"):
            # 准确率，求和计算算数平均值
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            print("+++++++++++++++")
            print(correct_predictions)
            print("+++++++++++++++")
            #将correct_predictions的数据格式转化成float.
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            print(self.accuracy)
