import os
import numpy as np
import datetime
import tensorflow as tf
from data_helper import *
import csv

# State which model to use here
from vdcnn import VDCNN

# Parameters settings
# Data loading params
tf.flags.DEFINE_string("database_path", "data/gaozhong/gaozhong_english/", "Path for the dataset to be used.")

# Model Hyperparameters
tf.flags.DEFINE_integer("sequence_max_length", 1024, "Sequence Max Length (default: 1024)")
tf.flags.DEFINE_string("downsampling_type", "maxpool", "Types of downsampling methods, use either three of maxpool, k-maxpool and linear (default: 'maxpool')")
tf.flags.DEFINE_integer("depth", 29, "Depth for VDCNN, use either 9, 17, 29 or 47 (default: 9)")
tf.flags.DEFINE_boolean("use_he_uniform", True, "Initialize embedding lookup with he_uniform (default: True)")
tf.flags.DEFINE_boolean("optional_shortcut", False, "Use optional shortcut (default: False)")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 1e-3, "Starter Learning Rate (default: 1e-2)")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("evaluate_every", 150, "Evaluate model on dev set after this many steps (default: 50)")
tf.flags.DEFINE_boolean("enable_tensorboard", True, "Enable Tensorboard (default: True)")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
print("Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr, value))
print("")

# Data Preparation
# Load data
print("Loading data...")
data_helper = data_helper(sequence_max_length=FLAGS.sequence_max_length)
train_data, train_label, test_data, test_label = data_helper.load_dataset(FLAGS.database_path)

print(train_data)
print(train_label)
print(train_data.shape)
print(train_label.shape)



num_batches_per_epoch = int((len(train_data)-1)/FLAGS.batch_size) + 1
print("Loading data succees...")

# ConvNet
acc_list = [0]
sess = tf.Session()
cnn = VDCNN(num_classes=train_label.shape[1], 
	depth=FLAGS.depth,
	sequence_max_length=FLAGS.sequence_max_length, 
	downsampling_type=FLAGS.downsampling_type,
	use_he_uniform=FLAGS.use_he_uniform,
	optional_shortcut=FLAGS.optional_shortcut)

# Optimizer and LR Decay
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	global_step = tf.Variable(0, name="global_step", trainable=False)
	learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.num_epochs*num_batches_per_epoch, 0.95, staircase=True)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	gradients, variables = zip(*optimizer.compute_gradients(cnn.loss))
	gradients, _ = tf.clip_by_global_norm(gradients, 7.0)
	train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

# Initialize Graph
sess.run(tf.global_variables_initializer())

# Train Step and Test Step
def train_step(x_batch, y_batch):
	"""
	A single training step
	"""
	feed_dict = {cnn.input_x: x_batch, 
				 cnn.input_y: y_batch, 
				 cnn.is_training: True}
	_, step, loss, fc3 = sess.run([train_op, global_step, cnn.loss, cnn.fc3], feed_dict)
	time_str = datetime.datetime.now().isoformat()
	pre, rec, pre_2, rec_2, pre_3, rec_3 = precision_recall(fc3, y_batch)
	print("{}: Step {}, Epoch {}, Loss {:g}".format(time_str, step, int(step//num_batches_per_epoch)+1, loss))
	print("1 {} {}, 2 {} {}, 3 {} {}".format(pre, rec, pre_2, rec_2, pre_3, rec_3))
	#if step%FLAGS.evaluate_every == 0 and FLAGS.enable_tensorboard:
	#	summaries = sess.run(train_summary_op, feed_dict)
	#	train_summary_writer.add_summary(summaries, global_step=step)

def test_step(x_batch, y_batch):
	"""
	Evaluates model on a dev set
	"""
	feed_dict = {cnn.input_x: x_batch, 
				 cnn.input_y: y_batch, 
				 cnn.is_training: True}
	loss, fc3 = sess.run([cnn.loss, cnn.fc3], feed_dict)
	pre, rec, pre_2, rec_2, pre_3, rec_3 = precision_recall(fc3, y_batch)
	return pre, rec, pre_2, rec_2, pre_3, rec_3, loss


def precision_recall(scores, y_batch):        
#    print(scores.shape)
#    print(len(y_batch))
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
# =============================================================================
#         if true_total_count != pred_total_count:
#             print("=============")
#             print(intersec_true)
#             print("=============")
# =============================================================================
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
#        print(y_indices[i])
        intersec_true = 0
        for j in y_indices_3[i]:
            intersec_true += y_batch[i][j]
#            print(intersec_true)
        true_total_count = np.count_nonzero(y_batch[i] == 1)
        pred_total_count = len(y_indices_3[i])
        pre_3 += intersec_true*1.0/pred_total_count
        rec_3 += intersec_true*1.0/true_total_count
    pre_3 = pre_3/len(y_batch)
    rec_3 = rec_3/len(y_batch)
    
    print("============")
    print(rec_3)
#    print(len(y_batch))
    return pre, rec, pre_2, rec_2, pre_3, rec_3






# Generate batches
train_batches = data_helper.batch_iter(list(zip(train_data, train_label)), FLAGS.batch_size, FLAGS.num_epochs)

# Training loop. For each batch...
for train_batch in train_batches:
	x_batch, y_batch = zip(*train_batch)
	train_step(x_batch, y_batch)
	current_step = tf.train.global_step(sess, global_step)
	# Testing loop
	if current_step % FLAGS.evaluate_every == 0:
		print("\nEvaluation:")
		i = 0
		sum_loss = 0
		sum_pre = 0
		sum_rec = 0
		sum_pre_2 = 0
		sum_rec_2 = 0
		sum_pre_3 = 0
		sum_rec_3 = 0       
		test_batches = data_helper.dev_batch_iter(list(zip(test_data, test_label)), FLAGS.batch_size, 1)
		y_preds = np.ones(shape=len(test_label), dtype=np.int)
		for test_batch in test_batches:
			x_test_batch, y_test_batch = zip(*test_batch)
			pre, rec, pre_2, rec_2, pre_3, rec_3, test_loss = test_step(x_test_batch, y_test_batch)
#			print(i)
#			print(pre, rec, pre_2, rec_2, pre_3, rec_3, test_loss)
			sum_loss += test_loss
			sum_pre += pre
			sum_rec += rec
			sum_pre_2 += pre_2
			sum_rec_2 += rec_2
			sum_pre_3 += pre_3
			sum_rec_3 += rec_3   
			i += 1
#			print("Evaluation Batch {:g}".format(i))
#			print("pre_3 {}, rec_3 {:g}".format(pre_3, rec_3))  
		sum_pre = sum_pre*128/1760
		sum_rec = sum_rec*128/1760
		sum_pre_2 = sum_pre_2*128/1760
		sum_rec_2 = sum_rec_2*128/1760
		sum_pre_3 = sum_pre_3*128/1760
		sum_rec_3 = sum_rec_3*128/1760

		time_str = datetime.datetime.now().isoformat()
		print("{}: Evaluation Summary, Loss {:g}".format(time_str, sum_loss/i))
		print("pre_1 {}, rec_1 {:g}".format(sum_pre, sum_rec))
		print("pre_2 {}, rec_2 {:g}".format(sum_pre_2, sum_rec_2))
		print("pre_3 {}, rec_3 {:g}".format(sum_pre_3, sum_rec_3))  
        
        
        
		writeFile = open(".\\data\\gaozhong\\gaozhong_english\\test_data_english_eval.csv",'a+',newline="")
		writer = csv.writer(writeFile)
		writer.writerow([current_step, sum_loss/i, sum_pre, sum_rec, sum_pre_2, sum_rec_2, sum_pre_3, sum_rec_3])
		writeFile.close()
        