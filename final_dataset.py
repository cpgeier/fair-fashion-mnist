# Converts dataset pickle files to the idx3-ubyte data format 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time
import pickle
import sys
import os
from PIL import Image
from array import *
from random import shuffle

def load_pickle(filename):
    pickle_out = open(filename, "rb")
    obj = pickle.load(pickle_out)
    return obj
loc = "data/"
final_loc = "final_dataset/"
x_train = load_pickle(loc + "x_train.pickle")
y_train = load_pickle(loc + "y_train.pickle")
x_test = load_pickle(loc + "x_test.pickle")
y_test = load_pickle(loc + "y_test.pickle")

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

print(type(x_train))
print(x_train.shape)


width, height = 28, 28

# Save training dataset

data_image = array('B')
data_label = array('B')

for i in range(len(x_train)):
    for x in range(0,28):
        for y in range(0,28):
            data_image.append(x_train[i][y][x])

    data_label.append(y_train[i])

hexval = "{0:#0{1}x}".format(len(x_train),6) 
print(hexval)
header = array('B')
header.extend([0,0,8,1,0,0])
header.append(int('0x'+hexval[2:][:2],16))
header.append(int('0x'+hexval[2:][2:],16))

data_label = header + data_label
if max([width,height]) <= 256:
    header.extend([0,0,0,width,0,0,0,height])
header[3] = 3

data_image = header + data_image

output_file = open(final_loc + 'train'+'-images-idx3-ubyte', 'wb')
data_image.tofile(output_file)
output_file.close()

output_file = open(final_loc + 'train'+'-labels-idx1-ubyte', 'wb')
data_label.tofile(output_file)
output_file.close()

# Save testing dataset

data_image = array('B')
data_label = array('B')

for i in range(len(x_test)):
    for x in range(0,width):
        for y in range(0,height):
            data_image.append(x_test[i][y][x])

    data_label.append(y_test[i])

hexval = "{0:#0{1}x}".format(len(x_test),6)
print(hexval)
header = array('B')
header.extend([0,0,8,1,0,0])
header.append(int('0x'+hexval[2:][:2],16))
header.append(int('0x'+hexval[2:][2:],16))

data_label = header + data_label
if max([width,height]) <= 256:
    header.extend([0,0,0,width,0,0,0,height])

header[3] = 3

data_image = header + data_image

output_file = open(final_loc + 't10k'+'-images-idx3-ubyte', 'wb')
data_image.tofile(output_file)
output_file.close()

output_file = open(final_loc + 't10k'+'-labels-idx1-ubyte', 'wb')
data_label.tofile(output_file)
output_file.close()

os.system('gzip '+ final_loc + 'train' +'-images-idx3-ubyte')
os.system('gzip '+ final_loc + 'train' +'-labels-idx1-ubyte')
os.system('gzip '+ final_loc + 't10k' +'-images-idx3-ubyte')
os.system('gzip '+ final_loc + 't10k' +'-labels-idx1-ubyte')

# The parts of the above numpy to idx-ubyte conversion comes from https://github.com/gskielian/JPG-PNG-to-MNIST-NN-Format
# The license for this modified code can be found at https://github.com/gskielian/JPG-PNG-to-MNIST-NN-Format/blob/master/LICENSE