import sys
import cmd
from import_mnist import import_mnist, display_mnist, input_image
from training_evaluation import train, test_pb, test_ckpt, test, inference
from import_png import parse_png
import os

# Import modes
training = 0
testing = 1

global mnist_training
	
class Shell(cmd.Cmd):
    intro = 'Convolutional neural network for handwriting recognition (Team 2).   Type help or ? to list commands.\n'
    prompt = 'Command? '
    file = None

    def do_get(self, arg):
        'Fetch MNIST dataset'
        global mnist_training
        global mnist_test
        'Get MNIST database from Internet'
        if arg=="train":
            print ("> Getting MNIST training set... ", end = '')
            sys.stdout.flush()
            mnist_training = import_mnist(training)
            print("> Imported training set of size: {}".format(len(mnist_training[1])))
        elif arg=="test":
            print ("> Getting MNIST testing set... ", end = '')
            sys.stdout.flush()
            mnist_test = import_mnist(testing)	
            print("> Imported testing set of size: {}".format(len(mnist_test[1])))
        else:
            print("> Wrong argument, please use 'get train' or 'get test'")
    def do_train(self, arg):
        'Train machine learning model'
        global mnist_training
        train(mnist_training[0], mnist_training[1])	
    def do_test_pb(self, arg):
        'Test .pb model'
        global mnist_test
        test_pb(mnist_test[0], mnist_test[1], int(arg))
    def do_test_ckpt(self):
        'Test .ckpt model'
        global mnist_test
        test_ckpt(mnist_test[0], mnist_test[1])
    def do_test(self, arg):
        global mnist_test
        accuracy = 0
        arg1, arg2 = arg.split()
        if arg1=="all":
            for i in range(len(mnist_test[1])):
                accuracy+=test(mnist_test[0], mnist_test[1], i, out = False)
            accuracy /= len(mnist_test[1])
            print(accuracy)
        elif arg1 == "image":
            image = [parse_png('./images/' + arg2, 28)]
            img = display_mnist(image)
            img.show()
            accuracy=inference(image)
        else:
            accuracy=test(mnist_test[0], mnist_test[1], int(arg), out = True)  
    def do_show(self, arg):
        'Show a specific image from database'
        global mnist_training
        input_image(mnist_test, int(arg))

if __name__ == '__main__':
    Shell().cmdloop()