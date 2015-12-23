import os
import os.path
import scipy.io as matio
from keras.utils import np_utils
import numpy as np
import cv2
import math
def read_data(path = "./avletters/Lips/", test_split = 0.2, data_split = 1, seed=113):

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
                seq_total_num += int(seq_size[2])

                if height == 0:
                    height = int(seq_size[0])
                    width = int(seq_size[1])
                if height > 0 and height != seq_size[0]:
                    print 'image size should be equal'
                    return 0

                label += [(ord(name[0])-ord('A'))]*seq_size[2]


    data = np.zeros((seq_total_num, 1, height, width))  #channel =1

    ind = 0
    for item in filespath:
        file = matio.loadmat(item)
        seq_size = file['siz'][0]
        temp_data = file['vid']
        seq_num = int(seq_size[2])
        # store every slice into an array
        for slice_ind in range(seq_num):
            slice = temp_data[:, slice_ind]
            slice = slice.reshape((width, height))
            data[ind, 0, :, :] = np.transpose(slice)
            ind += 1



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

def read_data_sequence(path = "./avletters/Lips/", test_split = 0.2, data_split = 1, sampling_step=3,seed = 113):
    filespath = []
    height = 0
    width = 0
    seq_total_num = 0       # total slice number
    label = []
    max_len = 0
    #For each directory in the tree rooted at directory top (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames)
    for root, dirs, files in os.walk(path):
        for name in files:
            if not name.startswith("."):
                filespath.append(root + name)
                file = matio.loadmat(root + name)
                seq_size = file['siz'][0]
                if seq_size[2]>max_len:
                    max_len = seq_size[2]
                seq_total_num += sampling_step

                if height == 0:
                    height = int(seq_size[0])
                    width = int(seq_size[1])
                if height > 0 and height != seq_size[0]:
                    print 'image size should be equal'
                    return 0
                label += [(ord(name[0])-ord('A'))]*sampling_step

    data = np.zeros((seq_total_num,int(max_len/sampling_step)+1, height, width))

    # cv2.namedWindow('test')
    sample_index = 0
    for item in filespath:
        file = matio.loadmat(item)
        seq_size = file['siz'][0]
        temp_data = file['vid']
        seq_length = int(seq_size[2])
        # store every slice into an array
        for step_i in range(sampling_step):
            for slice_ind in range(int(max_len)):
                if (slice_ind*sampling_step+step_i) <seq_length:
                    slice = temp_data[:, slice_ind*sampling_step+step_i]
                    slice = slice.reshape((width, height))
                    data[sample_index, slice_ind, :, :] = np.transpose(slice)
                    # cv2.imshow('test',data[sample_index, slice_ind, :, :]/255)
                    # print "sample_index=", sample_index, ";", "slice_ind=", slice_ind, \
                    #     "label=", label[sample_index], "\n"
                    #
                    # cv2.waitKey()
            sample_index+=1


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
    Y_train = np_utils.to_categorical(Y_train, 26)
    Y_test = np_utils.to_categorical(Y_test, 26)

    return (X_train, Y_train), (X_test, Y_test)