import sys
import tensorflow as tf
import numpy as np
import cmd, sys
import itertools
import threading
import time
from numpy import zeros, uint8, float32
from PIL import Image
from import_mnist import import_mnist, display_mnist, input_image
from training_evaluation import train, test_model

# Default input data
learning_rate_d = 0.0001 
epochs_d = 100
dropout_d = 0.1

# import modes
training = 0
testing = 1

done = False
global mnist_training

def animate_loading():
    global done
    for animation in itertools.cycle(['|', '/', '-', '\\']):
        if done == True:
            break
        sys.stdout.write('\b' + animation)
        sys.stdout.flush()
        time.sleep(0.1)
		
class Shell(cmd.Cmd):
    intro = 'Convolutional neural network for handwriting recognition (Team 2).   Type help or ? to list commands.\n'
    prompt = 'Command? '
    file = None

    # Commands
    def do_set_epoch(self, arg):
        'Set number of iterations'
        global iterations
        print ("\tSet number of iterations for training (Default:", iterations_d,"\b): ", end ='')
        iterations = int(input())
    def do_set_train_size(self, arg):
        'Set train size'
        global train_size
        print ("\tSet the number of images for training (Default:", train_size_d,"\b): ", end ='')
        train_size = int(input())
    def do_set_rate(self, arg):
        'Set learning rate'
        global rate
        print ("\tSet learning rate (Default:", rate_d,"\b): ", end ='')
        rate = float(input())
    def do_set_dropout(self, arg):
        'Set dropout probability'
    def do_set_default(self, arg):
        'Show all parameter values'
        print ("\tNumber of iterations:", iterations)
        print ("\tBatch size:", batch_size)
        print ("\tLearning rate:", rate)
        print ("\tNumber of classes:", n_classes)
    def do_get_MNIST_train(self, arg):
        'Fetch MNIST training dataset'
        global mnist_training
        'Get MNIST database from Internet'
        print ("> Getting MNIST training set... ", end = '')
        sys.stdout.flush()
        mnist_training = import_mnist(training)
        print("> Imported training set of size: {}".format(len(mnist_training[1])))
    def do_get_MNIST_test(self, arg):
        'Fetch MNIST testing dataset'
        global mnist_test
        print ("> Getting MNIST testing set... ", end = '')
        sys.stdout.flush()
        mnist_test = import_mnist(testing)	
        print("> Imported testing set of size: {}".format(len(mnist_test[1])))
    def do_show(self, arg):
        'Show a specific image from database'
        global mnist_training
        input_image(mnist_test)	
    def do_train(self, arg):
        'Train machine learning model'
        global mnist_training
        train(mnist_training[0], mnist_training[1])	
    def do_test(self, arg):
        'Test machine learning model'
        global mnist_test
        test_model(mnist_test[0], mnist_test[1])

def parse(arg):
    'Convert a series of zero or more numbers to an argument tuple'
    return tuple(map(int, arg.split()))

if __name__ == '__main__':
    Shell().cmdloop()