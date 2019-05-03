# Removes labeled duplicate images from dataset and outputs dataset as pickle files

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time
import pickle
import sys

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
inds = y_train.argsort()
sorted_x_train = x_train[inds]
sorted_y_train = np.sort(y_train)

inds = y_test.argsort()
sorted_x_test = x_test[inds]
sorted_y_test = np.sort(y_test)
dups = []
for i in range(0,10):
    pickle_in = open("label_dups_output/labeled_dups" + str(i) + ".pickle","rb")
    dups += pickle.load(pickle_in)

#remove_train = []
remove_test = []
#tracker_train = {}
tracker_test = {}
for i in range(0,10):
    #tracker_train[i] = 0
    tracker_test[i] = 0

for i in dups:
    if i[3] == 1:
        #remove_train.append(i[0])
        remove_test.append(i[1])
        #tracker_train[sorted_y_train[i[0]]] += 1
        tracker_test[sorted_y_test[i[1]]] += 1

#total_train_removed = 0
total_test_removed = 0
for i in range(0,10):
    print(i,":", tracker_test[i])
    #total_train_removed += tracker_train[i]
    total_test_removed += tracker_test[i]

#assert len(remove_train) == total_train_removed
assert len(remove_test) == total_test_removed
#assert total_test_removed == total_train_removed

#sorted_x_train = np.delete(sorted_x_train, remove_train,axis=0)
#sorted_y_train = np.delete(sorted_y_train, remove_train,axis=0)
sorted_x_test = np.delete(sorted_x_test, remove_test,axis=0)
sorted_y_test = np.delete(sorted_y_test, remove_test,axis=0)

print("Set lengths:")
print(len(sorted_x_train))
print(len(sorted_y_train))
print(len(sorted_x_test))
print(len(sorted_y_test))

#assert len(sorted_x_train)-len(remove_train) == len(sorted_x_train) - total_train_removed
assert len(sorted_x_test)-len(remove_test) == len(sorted_x_test) - total_test_removed

def write_pickle(filename, obj):
    pickle_out = open(filename, "wb")
    pickle.dump(obj, pickle_out)

assert len(sorted_x_train) == len(sorted_y_train)
assert len(sorted_x_test) == len(sorted_y_test)

train_random_p = np.random.permutation(len(sorted_x_train))
test_random_p = np.random.permutation(len(sorted_x_test))

sorted_x_train = sorted_x_train[train_random_p] 
sorted_y_train = sorted_y_train[train_random_p]
sorted_x_test = sorted_x_test[test_random_p]
sorted_y_test = sorted_y_test[test_random_p]

write_pickle("data/x_train.pickle",sorted_x_train)
write_pickle("data/y_train.pickle",sorted_y_train)
write_pickle("data/x_test.pickle",sorted_x_test)
write_pickle("data/y_test.pickle",sorted_y_test)