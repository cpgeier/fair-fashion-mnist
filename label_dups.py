# Provides GUI to label duplicate images in dataset and saves labels to pickle files

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time
import pickle
import sys
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

def identify_dups(filename):
    inds = y_train.argsort()
    sorted_x_train = x_train[inds]
    sorted_y_train = np.sort(y_train)

    inds = y_test.argsort()
    sorted_x_test = x_test[inds]
    sorted_y_test = np.sort(y_test)

    # Assumed to be 1 class per file

    #dup_file = input("Duplicates file name: ")
    pickle_in = open(filename,"rb")
    dups = pickle.load(pickle_in)

    num_non_duplicates = 0
    fin_dups = []

    def limit_reached():
        print("Writing to file...")
        pickle_out = open("label_dups_output/labeled_" + filename,"wb")
        pickle.dump(fin_dups, pickle_out)
        plt.close('all')
        print("Please press enter to continue.")

    class Index(object):
        ind = 0
        num_non_duplicates = 0

        def iden_duplicate(self, event):
            dups[self.ind].append(1)
            fin_dups.append(dups[self.ind])
            self.ind += 1
            val = dups[self.ind]
            img_fig_1.set_data(sorted_x_train[val[0]])
            img_fig_2.set_data(sorted_x_test[val[1]])
            plt.draw()

        def iden_non_duplicate(self, event):
            dups[self.ind].append(0)
            fin_dups.append(dups[self.ind])
            self.ind += 1
            val = dups[self.ind]
            img_fig_1.set_data(sorted_x_train[val[0]])
            img_fig_2.set_data(sorted_x_test[val[1]])
            self.num_non_duplicates += 1
            #print("Non dup.")
            if self.num_non_duplicates >20:
                limit_reached()
            plt.draw()

    img_1 = sorted_x_train[dups[0][0]]
    img_2 = sorted_x_test[dups[0][1]]
    f, axarr = plt.subplots(1,2,figsize=(15,10))
    axarr[0].title.set_fontsize(24)
    axarr[0].title.set_text("Image from Training Set")
    axarr[0].grid(False)
    img_fig_1 = axarr[0].imshow(img_1, cmap='gray_r')
    axarr[1].title.set_fontsize(24)
    axarr[1].title.set_text("Image from Testing Set")
    axarr[1].grid(False)
    img_fig_2 = axarr[1].imshow(img_2, cmap='gray_r')

    callback = Index()
    axprev = plt.axes([0.3, 0.05, 0.2, 0.075])
    axnext = plt.axes([0.51, 0.05, 0.2, 0.075])
    bnext = Button(axnext, 'Very Similar')
    bnext.label.set_fontsize(24)
    bnext.on_clicked(callback.iden_duplicate)
    bprev = Button(axprev, 'Distinct')
    bprev.label.set_fontsize(24)
    bprev.on_clicked(callback.iden_non_duplicate)
    #print("Showing plot")
    plt.show(block=False)
    plt.pause(.01)
    input("")
    #print("After Showing plot")

for i in range(0,10):
    fn = "dups/dups" + str(i) + ".pickle"
    print("Reading in ", fn, ". This could take a few seconds.")
    identify_dups(fn)