import os
import os.path

import scipy.io as matio
import numpy

import cv2 as cv


def read_avletter(path):

first_sequence = matio.loadmat('/Users/cfchen/Downloads/avletters/Lips/A1_Anya-lips.mat')
index=0
a =  os.walk('lip_reading')
#For each directory in the tree rooted at directory top (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames)
for i in a:
		for file_name in i[2]:
			cc = 1
			print (file_name)

#assign 'vid' element in "dict" structure to data
data = first_sequence['vid']
first_img = data[:,1]
first_img = first_img.reshape((80,60))


first_img = np.transpose(first_img)

cv.namedWindow("test",0)
cv.imshow("test",first_img)
cv.waitKey()
cv.destroyWindow("test")
batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

first_img = X_train[0,0,:,:]












