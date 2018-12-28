import time
import datetime
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from convolutional_net import maxpool, conv_net, weights, biases
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

def next_batch(images, labels, batch_size, offset):
	batch_x = images[offset:offset+batch_size]
	batch_y = labels[offset:offset+batch_size]
	return batch_x, batch_y
	
def quantizeData(data):
    ## you can design your own quantization method
    ## the following method is a naive implementation and it quantizes the numbers to -1 or +1.
    quantizedData = np.sign(data)
    return quantizedData
	
def count_params():
    total_parameters = 0
    # Iterate over all variables
    for variable in tf.trainable_variables(): 
        #print(variable.name)
        local_parameters=1
		# Get shape of the variable
        shape = variable.get_shape()  
        for i in shape:
			# multiply dimension values
            local_parameters*=i.value  
        total_parameters+=local_parameters
    print("> Model size: %d parameters" %total_parameters)
	
def inference(image):
	# Set up directory of model
	model_directory = os.path.join(os.getcwd(), 'model')
	output_converted_graph_name = os.path.join(model_directory, 'model_converted.tflite')
	
	# Load TFLite model and allocate tensors.
	interpreter = tf.contrib.lite.Interpreter(model_path=output_converted_graph_name)
	interpreter.allocate_tensors()

	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	# Test model with testing dataset
	input_shape = input_details[0]['shape']
	input_data = image
	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()
	output_data = interpreter.get_tensor(output_details[0]['index'])
	
	# Output prediction with highest probability
	max_pred_index = np.argmax(output_data[0])
	print("> Predicted value is", max_pred_index)
	print("> Probability:", output_data[0][max_pred_index])


	
def test(test_images_in, test_labels_in, index, out):
	# Set up directory of model
	model_directory = os.path.join(os.getcwd(), 'model')
	output_converted_graph_name = os.path.join(model_directory, 'model_converted.tflite')
	
	# Load TFLite model and allocate tensors.
	interpreter = tf.contrib.lite.Interpreter(model_path=output_converted_graph_name)
	interpreter.allocate_tensors()

	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	# Test model with testing dataset
	input_shape = input_details[0]['shape']
	input_data = test_images_in[index:index+1]
	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()
	output_data = interpreter.get_tensor(output_details[0]['index'])
	
	# Output prediction with highest probability
	max_pred = max(output_data[0])
	max_pred_index = np.argmax(output_data[0])
	max_truth_index = np.argmax(test_labels_in[index])
	if out == True:
		print ("> Real value is '", max_truth_index, "'")
		print("> Predicted value is '", max_pred_index, "'(prob. of", max_pred, ")")
	if max_pred_index == max_truth_index:
		return 1
	else:
		return 0
	
def train(train_images_in, train_labels_in):
	# Initialize variables
	num_epochs = 50
	batch_size = 100
	learning_rate = 0.0001 
	# Remove TF warnings	
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	# Create new log file
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
			# Initialization of placeholders		
			images_unshaped = tf.placeholder(tf.float32, shape=[None, 28*28], name='images_unshaped')
			images = tf.reshape(images_unshaped, [-1, 28, 28, 1])
			labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
			labels_classes = tf.argmax(labels, axis=1)
			sess.run(tf.global_variables_initializer())
			
			# Define the model
			conv_out = conv_net(images)
			conv_out_norm = tf.nn.softmax(conv_out, name="conv_out_norm")
			conv_out_classes = tf.argmax(conv_out_norm, axis=1)	
			
			# Define cost function and optimizer
			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=conv_out, labels=labels))
			optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy)
			
			# Define prediction functions
			correct_prediction = tf.equal(conv_out_classes, labels_classes)
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
			
			# Split training dataset in training and validation (55000, 5000)
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
			
			# Initialize all variables
			sess.run(tf.global_variables_initializer())
			
			# Start training
			log = "> Training..."
			f.write("%s\t%s\r\n"% (datetime.datetime.now(), log))
			print(log)
			sess_start_t =time.time()
			for epoch in range(num_epochs):
				epoch_start_t = time.time()
				train_accuracy = 0
				offset = 0
				with tf.device("/gpu:0"):
					for batch in range(0, int(len(train_labels)/batch_size)):	
						# Load training images and labels in batches
						batch_x, batch_y = next_batch(train_images, train_labels, batch_size, offset)
						sess.run(optimizer, feed_dict={images_unshaped: batch_x, labels: batch_y})				
						train_accuracy += sess.run(accuracy, feed_dict={images_unshaped: batch_x, labels: batch_y}) 
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
			log = "> Session completed! Total duration:\t"+str(int(sess_end_t-sess_start_t))+" seconds"
			f.write("%s\t%s\r\n"% (datetime.datetime.now(), log))
			print(log)
			
			# Save the graph
			saver = tf.train.Saver()
			model_directory = os.path.join(os.getcwd(), 'model')
			tf.train.write_graph(sess.graph_def, model_directory, 'savegraph.pbtxt')
			saver.save(sess, os.path.join(model_directory, 'model.ckpt'))
			print('> Graph saved!')

			# Freeze the graph
			input_graph_path = os.path.join(model_directory, 'savegraph.pbtxt')
			checkpoint_path = os.path.join(model_directory, 'model.ckpt')
			input_saver_def_path = ""
			input_binary = False
			input_node_names = "images_unshaped"
			output_node_names = "conv_out_norm"	
			input_nodes = images_unshaped
			output_nodes = conv_out_norm
			restore_op_name = "save/restore_all"
			filename_tensor_name = "save/Const:0"
			output_frozen_graph_name = os.path.join(model_directory, 'model_frozen.pb')

			clear_devices = True
			freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")			
			print('> Graph freezed!')
			
			# Optimize the graph
			output_optimized_graph_name = os.path.join(model_directory, 'model_optimized.pb')
			input_graph_def = tf.GraphDef()
			with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
				data = f.read()
				input_graph_def.ParseFromString(data)
			output_graph_def = optimize_for_inference_lib.optimize_for_inference(
				input_graph_def,
				[input_node_names], # an array of the input node(s)
				[output_node_names], # an array of output nodes
				tf.float32.as_datatype_enum)
			f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
			f.write(output_graph_def.SerializeToString()) # save optimized graph
			print('> Graph optimized!')

			# Convert the graph
			output_converted_graph_name = os.path.join(model_directory, 'model_converted.tflite')
			input_graph_def = tf.GraphDef()
			with tf.gfile.Open(output_optimized_graph_name, "rb") as f:
				data = f.read()
				input_graph_def.ParseFromString(data)
			converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
				output_optimized_graph_name, [input_node_names], [output_node_names], input_shapes={	"images_unshaped": [None, 28*28]})
			converter.post_training_quantize = True	
			tflite_model = converter.convert()
			open(output_converted_graph_name, "wb").write(tflite_model)	
			print('> Graph converted!')
			
			sess.close()
			
def test_pb(test_images_in, test_labels_in, index):
	model_directory = os.path.join(os.getcwd(), 'model')
	output_optimized_graph_name = os.path.join(model_directory, 'model_optimized.pb')
	with tf.gfile.GFile(output_optimized_graph_name, "rb") as f:
		restored_graph_def = tf.GraphDef()
		restored_graph_def.ParseFromString(f.read())		
	with tf.Graph().as_default() as graph:
		tf.import_graph_def(
			restored_graph_def,
			input_map=None,
			return_elements=None,
			name="")	
		# Show all available parameters
		#for op in tf.get_default_graph().get_operations():
		#	print(str(op.name)) 		
		# Get necessary tensors by name
		images_unshaped_tensor = graph.get_tensor_by_name("images_unshaped:0")
		prediction = graph.get_tensor_by_name("conv_out_norm:0")		
		sess=tf.Session(graph=graph)	
		count_params()
		
		# Load one image from testing set
		test_images = test_images_in[index:index+1]	
		
		# Feed testing images to input placeholder and calculate prediction
		result=sess.run(prediction, feed_dict={images_unshaped_tensor: test_images})
		print(result)
		
def test_ckpt(test_images_in, test_labels_in):
    # create an empty graph for the session
	tf.reset_default_graph()
	saver = tf.train.import_meta_graph("model.meta")  
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	with tf.Session() as sess:
		# Restore previous graph
		saver.restore(sess, os.path.join(os.getcwd(), 'model'))
		graph = tf.get_default_graph()
		count_params()
		
		# Get necessary tensors by name
		images_unshaped_tensor = graph.get_tensor_by_name("images_unshaped:0")
		labels_tensor = graph.get_tensor_by_name("labels:0")
		accuracy_tensor = graph.get_tensor_by_name("accuracy:0")

		# Initialize variables
		batch_size_test = 100
		offset = 0
		test_accuracy = 0
		
		# Make prediction
		for batch in range(0, int(len(test_labels_in)/batch_size_test)):
			# Load test images and labels in batches
			test_images, test_labels = next_batch(test_images_in, test_labels_in, batch_size_test, offset)
			test_accuracy += sess.run(accuracy_tensor, feed_dict={images_unshaped_tensor:test_images, labels_tensor:test_labels})
			offset = offset + batch_size_test
		test_accuracy /= int(len(test_labels_in)/batch_size_test)	
		log = ">\t- Testing Accuracy:\t{}".format(test_accuracy)
		print(log)
				