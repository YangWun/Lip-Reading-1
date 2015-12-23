import numpy as np
np.random.seed(1337)  # for reproducibility

#import layers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, AutoEncoder,Permute,Reshape
from keras.layers.core import ActivityRegularization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import containers
from keras.regularizers import l2
from keras.layers.convolutional import Convolution1D

#import visualization lib
from keras.utils.visualize_util import plot
import theano



import os
import os.path
import scipy.io as matio
import cv2
from read_data import read_data
def read_data_motion(path = "./avletters/Lips/", test_split = 0.2, data_split = 1, seed=113):

    filespath = []
    height = 0
    width = 0
    seq_total_num = 0       # total slice number

    label = []
    #For each directory in the tree rooted at directory top (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames)
    for root, dirs, files in os.walk(path):
        for name in files:
            if not name.startswith("."):
                filespath.append(root + name)
                file = matio.loadmat(root + name)
                seq_size = file['siz'][0]
                seq_total_num += int(seq_size[2])-1

                if height == 0:
                    height = int(seq_size[0])
                    width = int(seq_size[1])
                if height > 0 and height != seq_size[0]:
                    print 'image size should be equal'
                    return 0

                label += [(ord(name[0])-ord('A'))]*(seq_size[2]-1)

    data = np.zeros((seq_total_num, 1, height, width))  #channel =1

    ind = 0
    for item in filespath:
        file = matio.loadmat(item)
        seq_size = file['siz'][0]
        temp_data = file['vid']
        seq_num = int(seq_size[2])-1
        # store every slice into an array
        for slice_ind in range(seq_num):
            slice = temp_data[:, slice_ind+1]-temp_data[:, slice_ind]
            slice = slice.reshape((width, height))
            data[ind, 0, :, :] = np.transpose(slice)
            ind += 1

   # cv.namedWindow('test')
   # cv.imshow('test', np.uint8(data_out[0, :, :]))
   # cv.waitKey()
   # cv.destroyAllWindows()

    data = data.astype('float32')
    data /= 255

    np.random.seed(seed)
    np.random.shuffle(data)
    np.random.seed(seed)
    np.random.shuffle(label)

    l = int(len(data) * data_split)
    tl = int(l * (1 - test_split))

    X_train = data[:tl, :, :, :]
    Y_train = label[:tl]

    X_test = data[tl:l, :, :, :]
    Y_test = label[tl:l]

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train, 26)
    Y_test = np_utils.to_categorical(Y_test, 26)

    return (X_train, Y_train), (X_test, Y_test)