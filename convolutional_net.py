import tensorflow as tf

def conv2d(x, W, b):
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 
	
def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # Outputs random values from a truncated normal distribution
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) 
    return tf.Variable(initial)	
	
weights = {
    "w_conv1": weight_variable([5,5,1,32]),
    "w_conv2":  weight_variable([5,5,32,64]),
    "w_fconnected": weight_variable([7*7*64,1024]),
    "w_out":  weight_variable([1024,10])
}
biases = {
    "b_conv1": bias_variable([32]),
    "b_conv2": bias_variable([64]),
    "b_fconnected": bias_variable([1024]),
    "b_out": bias_variable([10])
}

def conv_net(x_in): 
    conv1 = conv2d(x_in, weights["w_conv1"], biases["b_conv1"])
    conv1_pool = maxpool(conv1)
    conv2 = conv2d(conv1_pool, weights["w_conv2"], biases["b_conv2"])
    conv2_pool = maxpool(conv2)
    fconnected_flat = tf.reshape(conv2_pool, [-1,7*7*64])
    fconnected = tf.nn.relu(tf.matmul(fconnected_flat, weights["w_fconnected"]) + biases["b_fconnected"])
    out = tf.matmul(fconnected, weights["w_out"]) + biases["b_out"]
    return out
	

	

