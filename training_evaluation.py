import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from convolutional_net import conv_net
import time
import datetime
import os
learning_rate = 0.0001 
# https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
#https://gist.github.com/SinghalHarsh/5d599cbfd38e7fcdb78d5c3802f12187#file-convolutional_neural_network-tensorflow-ipynb
#https://medium.com/data-science-group-iitr/building-a-convolutional-neural-network-in-python-with-tensorflow-d251c3ca8117
#https://stackoverflow.com/questions/51306862/how-to-use-tensorflow-gpu

def next_batch(images, labels, batch_size, offset):
	batch_x = images[offset:offset+batch_size]
	batch_y = labels[offset:offset+batch_size]
	return batch_x, batch_y
	
def test_model(test_images_in, test_labels_in):
    # create an empty graph for the session
	tf.reset_default_graph()
	saver = tf.train.import_meta_graph("model.meta")  
	
	with tf.Session() as sess:
		#restor previous graph
		saver.restore(sess, os.path.join(os.getcwd(), 'model'))
		graph = tf.get_default_graph()
		
		# get necessary tensors by name
		images_unshaped_tensor = graph.get_tensor_by_name("images_unshaped:0")
		labels_tensor = graph.get_tensor_by_name("labels:0")
		accuracy_tensor = graph.get_tensor_by_name("accuracy:0")

		# initialize variables
		batch_size_test = 100
		offset = 0
		test_accuracy = 0
		
		# make prediction
		for batch in range(0, int(len(test_labels_in)/batch_size_test)):
			# load test images and labels in batches
			test_images, test_labels = next_batch(test_images_in, test_labels_in, batch_size_test, offset)
			test_accuracy += sess.run(accuracy_tensor, feed_dict={images_unshaped_tensor:test_images, labels_tensor:test_labels})
			offset = offset + batch_size_test
		test_accuracy /= int(len(test_labels_in)/batch_size_test)	
		log = ">\t- Testing Accuracy:\t{}".format(test_accuracy)
		print(log)

		
def train(train_images_in, train_labels_in):	
	current_path = os.path.dirname(os.path.realpath(__file__))
	subdir_log = "Log_files"
	filename_log = "Log_training.txt"
	filepath_log = os.path.join(current_path, subdir_log, filename_log)
	if not os.path.exists(os.path.join(current_path, subdir_log)):
		os.mkdir(os.path.join(current_path, subdir_log))
	with open(filepath_log, 'w', encoding = 'utf-8') as f:
		log = "> Train session started"
		f.write("%s\t%s\r\n"% (datetime.datetime.now(), log))
		print(log)
		with tf.Session() as sess:
			# Placeholder variable for the input images /check
			images_unshaped = tf.placeholder(tf.float32, shape=[None, 28*28], name='images_unshaped')
			images = tf.reshape(images_unshaped, [-1, 28, 28, 1])
		
			# Placeholder variable for the true labels associated with the images /check
			labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
			labels_classes = tf.argmax(labels, axis=1)
		
			num_epochs = 100
			batch_size = 100
		
			conv_out = conv_net(images)
			
			# Use Softmax function to normalize the output /check
			conv_out_norm = tf.nn.softmax(conv_out)
			conv_out_classes = tf.argmax(conv_out_norm, axis=1)

			# Use Cross entropy cost function /check
			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=conv_out, labels=labels))
			
			# Use Adam Optimizer /check
			optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy)
			
			# Accuracy /check
			correct_prediction = tf.equal(conv_out_classes, labels_classes)
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
			log = "> Splitting dataset in training and validation datasets..."
			f.write("%s\t%s\r\n"% (datetime.datetime.now(), log))
			print(log)
			train_images, val_images, train_labels, val_labels = train_test_split(train_images_in, train_labels_in, test_size=0.08333)
			log = "> Training dataset size: {}".format(len(train_images))
			f.write("%s\t%s\r\n"% (datetime.datetime.now(), log))
			print(log)
			log = "> Validation dataset size: {}".format(len(val_images))
			f.write("%s\t%s\r\n"% (datetime.datetime.now(), log))
			print(log)
			# Initialize all variables /check
			sess.run(tf.global_variables_initializer())
			# Loop over number of epochs /check
			log = "> Training..."
			f.write("%s\t%s\r\n"% (datetime.datetime.now(), log))
			print(log)
			sess_start_t =time.time()
			for epoch in range(100):
				epoch_start_t = time.time()
				train_accuracy = 0
				offset = 0
				with tf.device("/gpu:0"):
					for batch in range(0, int(len(train_labels)/batch_size)):	
						# Generate a new batch of images (batch_x) and labels (batch_y) for training
						batch_x, batch_y = next_batch(train_images, train_labels, batch_size, offset)
						# Run the optimizer using loaded batch /check
						sess.run(optimizer, feed_dict={images_unshaped: batch_x, labels: batch_y})				
						# Calculate the accuracy on the batch of training data /check
						train_accuracy += sess.run(accuracy, feed_dict={images_unshaped: batch_x, labels: batch_y}) 
						# Generate offset for generation of next batch
						offset = offset + batch_size
				train_accuracy /= int(len(train_labels_in)/batch_size)
				vali_accuracy = sess.run(accuracy, feed_dict={images_unshaped:val_images, labels:val_labels})
				epoch_end_t = time.time()	
				log = "> Epoch {}".format(epoch+1)
				f.write("%s\t%s\r\n"% (datetime.datetime.now(), log))
				print(log)
				log = ">\t- Duration:\t"+str(int(epoch_end_t-epoch_start_t))+" seconds"
				f.write("%s\t%s\r\n"% (datetime.datetime.now(), log))
				print(log)
				log = ">\t- Training Accuracy:\t{}".format(train_accuracy)
				f.write("%s\t%s\r\n"% (datetime.datetime.now(), log))
				print(log)
				log = ">\t- Validation Accuracy:\t{}".format(vali_accuracy)
				f.write("%s\t%s\r\n"% (datetime.datetime.now(), log))
				print(log)
			sess_end_t =time.time()
			log = ">Session completed! Total duration:\t"+str(int(sess_end_t-sess_start_t))+" seconds"
			f.write("%s\t%s\r\n"% (datetime.datetime.now(), log))
			print(log)
			saver = tf.train.Saver()
			saver.save(sess, os.path.join(os.getcwd(), 'model'))