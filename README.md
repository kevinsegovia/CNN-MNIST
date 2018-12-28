# T2-CNN-MNIST

Course project from Advanced Topics in IC design ("Artificial Intelligence for Embedded Systems"), winter semester, Technical University of Munich.
The objective of this project is to cover the basics of deep learning models and its optimization by designing an artificial neural network for recognition of handwritten digits from MNIST database. The network is implemented on a RaspberryPi 3 B+.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
1. Installation of Tensorflow on RPi. Official tensorflow installation steps may not include neon dependencies in tensor_utils. This results in an exception as 'undefined symbol' when loading the tflite model. Instead, tensorflow should be installed as follows:

```
$ sudo apt-get install python-pip python3-pip
$ sudo pip3 uninstall tensorflow
$ git clone https://github.com/PINTO0309/Tensorflow-bin.git
$ cd Tensorflow-bin
$ sudo pip3 install tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl
```
2. Other modules like sklearn, PIL, numpy, gzip, etc.

## Running the tests
Once the repository is cloned, launch the command shell by executing
```
$ python3 ./main
```

### Training
First, fetch the MNIST training dataset by running the following command
```
get train
```
Then start the default training process:
```
train
```
When the training process is completed (50 epochs), it will be generated a "model" folder with the following files:
1. Checkpoint files .index, .meta and .ckpt
2. Frozen model without optimization: model_frozen.pb
3. Frozen model with optimization: model_optimized.pb
4. TFlite model with post-training quantization: model_conferted.tflite

### Test with MNIST dataset
First, fetch the MNIST test dataset by running the following command
```
get test
```
Then start the default testing process:
```
test [arg1]
```
Where [arg1] can be
  * *all*: will test the complete database and output the testing accuracy
  * *1-10000*: will test an specific sample from MNIST dataset


### Test with own image
Simply call the testing function:
```
test image [arg1]
```
Where [arg1] is the name of the .png image to be tested, inside thefolder ./images/. Example 
```
test image image_1.png
```
