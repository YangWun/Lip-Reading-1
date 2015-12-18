import numpy as np
np.random.seed(1337)  # for reproducibility

#import layers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, AutoEncoder
from keras.layers.core import ActivityRegularization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import containers
from keras.regularizers import l2


#import visualization lib
from keras.utils.visualize_util import plot
import theano








import os
import os.path
import scipy.io as matio
import cv2
from read_data import read_data

data_import = read_data('./avletters/Lips/');
first_sequence = matio.loadmat('./avletters/Lips/A1_Anya-lips.mat')

#assign 'vid' element in "dict" structure to data
data = first_sequence['vid']
first_img = data[:,1]
first_img = first_img.reshape((80,60))

batch_size = 10
nb_epoch = 12

# input image dimensions
# build teaching signal for auto-encoder




#aaaa = np.array([[2,3,4,5,6,7],[1,2,3,4,5,6]])
#test = aaaa[:,1]



a = range(100)
data_import = data_import[a,:,:]        # should use ndarray (np.array)
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
num_hidden = 100

#X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)


model = Sequential()


model.add(Dense(num_hidden,'glorot_uniform','linear',None,l2(0.01),None,None,None,None,80*60))
model.add(Dense(80*60,'glorot_uniform','linear',None,l2(0.01),None,None,None,None))





# encoder = containers.Sequential([Dense(num_sample, input_dim=num_dim), Dense(num_hidden,'glorot_uniform','linear',None,l2(0.01),None,None,None,None)])
# decoder = containers.Sequential([Dense(num_sample, input_dim=num_hidden), Dense(num_dim,'glorot_uniform','linear',None,l2(0.01),None,None,None,None)])
#
#
# model.add(AutoEncoder(encoder=encoder, decoder=decoder,output_reconstruction=True))


#model.add(AutoEncoder(encoder=encoder, decoder=decoder,output_reconstruction=True))
#model.add(ActivityRegularization(0.01,0.01))        #add regularization terms on activity?? or weight
model.compile(loss='mean_squared_error', optimizer='adadelta')





X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
Y_train = X_train

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(X_train, Y_train))




plot(model, to_file='model.png')


out_x = model.predict(X_train, 10, verbose=0)
for ii in range(out_x.shape[0]):
    out = out_x[ii][:]
    out = out.reshape(60,80)
    cv2.namedWindow('111')
    cv2.imshow('111',out)
    cv2.waitKey()
    cv2.destroyAllWindows()

