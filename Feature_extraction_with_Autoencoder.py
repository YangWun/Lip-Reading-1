from __future__ import print_function
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

data_import = read_data('./avletters/Lips/')

#assign 'vid' element in "dict" structure to data



batch_size = 128
nb_epoch = 1

# input image dimensions
# build teaching signal for auto-encoder

#aaaa = np.array([[2,3,4,5,6,7],[1,2,3,4,5,6]])
#test = aaaa[:,1]



# a = range(100)
# data_import = data_import[a,:,:]        # should use ndarray (np.array)
X_test = data_import
X_train = data_import
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


Y_train = X_train
Y_test = X_test

num_sample = X_train.shape[0]
num_dim = X_train.shape[1]*X_train.shape[2]
num_hidden = 256

#X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)


cv2.namedWindow('111')
for test_i in range(1000):
    first_img = X_train[test_i,:,:]
    cv2.imshow('111',first_img)
    cv2.waitKey()


model = Sequential()





model.add(Dense(num_hidden,'glorot_uniform','sigmoid',None,l2(0.00),None,None,None,None,80*60))
model.add(Dense(48,'glorot_uniform','sigmoid',None,l2(0.00),None,None,None,None))
model.add(Dense(80*60,'glorot_uniform','linear',None,l2(0.00),None,None,None,None))


model.compile(loss='mean_squared_error', optimizer='adadelta')



# encoder = containers.Sequential([Dense(num_sample, input_dim=num_dim), Dense(num_hidden,'glorot_uniform','linear',None,l2(0.01),None,None,None,None)])
# decoder = containers.Sequential([Dense(num_sample, input_dim=num_hidden), Dense(num_dim,'glorot_uniform','linear',None,l2(0.01),None,None,None,None)])
# model.add(AutoEncoder(encoder=encoder, decoder=decoder,output_reconstruction=True))


#model.add(AutoEncoder(encoder=encoder, decoder=decoder,output_reconstruction=True))
#model.add(ActivityRegularization(0.01,0.01))        #add regularization terms on activity?? or weight






X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
Y_train = X_train


# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#           show_accuracy=True, verbose=1, validation_data=(X_train, Y_train))


model.load_weights('./weight_file')

# plot(model, to_file='model.png')
# weight_out = model.layers[0].get_weights()
#
# weight_out_show = weight_out[0]
# weight_out_show = weight_out_show[:,0]
# weight_out_show = weight_out_show.reshape(60,80)
# cv2.namedWindow('111')
# cv2.imshow('111',weight_out_show)
# cv2.waitKey()
# cv2.destroyAllWindows()




out_x = model.predict(X_train, 128, verbose=0)
for ii in range(out_x.shape[0]):
    out = out_x[ii,:]
    out = out.reshape(60,80)
    cv2.imshow('111',out)
    cv2.waitKey()


















