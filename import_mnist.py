from struct import unpack
import sys
import itertools
import threading
import time
import gzip
import urllib.request
import os
import numpy as np
from numpy import zeros, uint8, float32, int32
from PIL import Image


images_cols = 0
images_rows = 0
training = 0
testing = 1

def import_mnist(mode):
	current_path = os.path.dirname(os.path.realpath(__file__))
	subdir_log = "Log_files"
	filename_log = "Log_import.txt"
	filepath_log = os.path.join(current_path, subdir_log, filename_log)
	if not os.path.exists(os.path.join(current_path, subdir_log)):
		os.mkdir(os.path.join(current_path, subdir_log))
	with open(filepath_log, 'w', encoding = 'utf-8') as f:
	
		# Download dataset files from MNIST website
		subdir_database = "Database_MNIST"
		if not os.path.exists(os.path.join(current_path, subdir_database)):
			os.mkdir(os.path.join(current_path, subdir_database))
			
		filename_images = "MNIST_train_images.gz"
		filepath_images = os.path.join(current_path, subdir_database, filename_images)
		filename_labels = "MNIST_train_labels.gz"
		filepath_labels = os.path.join(current_path, subdir_database, filename_labels)
		filename_images_test = "MNIST_test_images.gz"
		filepath_images_test = os.path.join(current_path, subdir_database, filename_images_test)
		filename_labels_test = "MNIST_test_labels.gz"
		filepath_labels_test = os.path.join(current_path, subdir_database, filename_labels_test)

		images_link = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
		labels_link = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
		images_test_link = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
		labels_test_link = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
		
		
		f.write("Dowloading images file from %s\r\n"% images_link)
		#urllib.request.urlretrieve(images_link, filepath_images)	
		f.write("Dowloading labels file from %s\r\n"% labels_link)
		#urllib.request.urlretrieve(labels_link, filepath_labels)	
		f.write("Dowloading labels file from %s\r\n"% images_test_link)
		#urllib.request.urlretrieve(images_test_link, filepath_images_test)
		f.write("Dowloading labels file from %s\r\n"% labels_test_link)
		#urllib.request.urlretrieve(labels_test_link, filepath_labels_test)	
		
		if(mode == training):
			# Read compressed file with gzip (binary mode) and close original file
			images = gzip.open(filepath_images, 'rb')
			f.write("Loading images file from %s\r\n"% filepath_images)
			labels = gzip.open(filepath_labels, 'rb')	
			f.write("Loading labels file from %s\r\n"% filepath_labels)	
		else:
			# Read compressed file with gzip (binary mode) and close original file
			images = gzip.open(filepath_images_test, 'rb')
			f.write("Loading images file from %s\r\n"% filepath_images_test)
			labels = gzip.open(filepath_labels_test, 'rb')	
			f.write("Loading labels file from %s\r\n"% filepath_labels_test)			
				
		# Read and verify magic number (images) 
		magic_number_images = images.read(4)
		magic_number_images = unpack('>I', magic_number_images)[0]
		f.write("magic_number_images = %s\r\n"% magic_number_images)
	
		# Read number of images, number of rows and number of columns
		images_number = images.read(4)
		images_number = unpack('>I', images_number)[0]
		f.write("images_number = %s\r\n"% images_number)
		images_rows = images.read(4)
		images_rows = unpack('>I', images_rows)[0]
		f.write("images_rows = %s\r\n"% images_rows)
		images_cols = images.read(4)
		images_cols = unpack('>I', images_cols)[0]
		f.write("images_cols = %s\r\n"% images_cols)
	
		# Read and verify magic number (labels) 
		magic_number_labels = labels.read(4)
		magic_number_labels = unpack('>I', magic_number_labels)[0]
		f.write("magic_number_labels = %s\r\n"% magic_number_labels)
		
		# Read number of items and verify
		labels_items = labels.read(4)
		labels_items = unpack('>I', labels_items)[0]
		f.write("labels_items = %s\r\n"% labels_items)	
		
		# Verify number of images match number of items (labels)
		if images_number != labels_items:
			f.write("Number of images does not match number of items (labels)\n")		
		# Fetch pixel values
		train_x = zeros((images_number, images_rows, images_cols), dtype=float32)
		y = zeros((images_number, 1), dtype=int32)
		train_x = zeros((images_number, images_rows * images_cols), dtype=float32)
		train_y = zeros((images_number, 10), dtype=int32)
		for i in range(images_number):
			for px in range(images_rows * images_cols):
				tmp_pixel = images.read(1)
				tmp_pixel = unpack('>B', tmp_pixel)[0]
				train_x[i][px] = tmp_pixel
			tmp_label = labels.read(1)
			y[i] = unpack('>B', tmp_label)[0]
			if y[i] == 0:
				train_y[i][0] = 1
			elif y[i] == 1:
				train_y[i][1] = 1
			elif y[i] == 2:
				train_y[i][2] = 1
			elif y[i] == 3:
				train_y[i][3] = 1
			elif y[i] == 4:
				train_y[i][4] = 1
			elif y[i] == 5:
				train_y[i][5] = 1
			elif y[i] == 6:
				train_y[i][6] = 1
			elif y[i] == 7:
				train_y[i][7] = 1
			elif y[i] == 8:
				train_y[i][8] = 1
			else:
				train_y[i][9] = 1				
		print("\bdone")
		return (train_x, train_y)	
		
def display_mnist(image_raw):
    image_2D = (np.reshape(image_raw, (28, 28))).astype(np.uint8)
    image_out = Image.fromarray(image_2D, 'L')
    return image_out	
	
def input_image(unpacked_images):
	image = int(input("\tType image to display (0 - 60000): "))
	img = display_mnist(unpacked_images[0][image])
	print ("\tLabel:", unpacked_images[1][image],"\b): ")
	img.show()

