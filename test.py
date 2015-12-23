import numpy as np
np.random.seed(1337)  # for reproducibility

#import layers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, AutoEncoder,Permute,Reshape, Masking,TimeDistributedDense
from keras.layers.core import ActivityRegularization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import containers
from keras.regularizers import l2, activity_l1
from keras.layers.convolutional import Convolution1D
from keras.layers.recurrent import LSTM


#import visualization lib
import theano


from keras.optimizers import RMSprop
import os
import os.path
import scipy.io as matio
import cv2

from read_data import read_data
from read_data import read_data_sequence

(data_import,data_label),(test_x,test_y) = read_data_sequence('../../Datasets/avletters/Lips/')
#assign 'vid' element in "dict" structure to data



batch_size = 128
nb_epoch = 1000




X_train = data_import
X_train = X_train.astype('float32')


num_hidden = 128



model = Sequential()
model_feature_extract = Sequential()



model.add(Dense(num_hidden,'glorot_uniform','sigmoid',None,l2(0.00),None,activity_l1(0.0),None,None,80*60))
model.add(Dense(80*60,'glorot_uniform','linear',None,l2(0.00),None,None,None,None))
model.compile(loss='mean_squared_error', optimizer='RMSprop')
model.load_weights('./weight_128')
auto_encoder_weight = model.layers[0].get_weights()


model_feature_extract.add(model.layers[0])
model_feature_extract.set_weights(auto_encoder_weight)
model_feature_extract.compile(loss='mean_squared_error', optimizer='RMSprop')


# cv2.namedWindow('test')
# for ii in range(X_train.shape[0]):
#     for jj in range(X_train.shape[1]):
#         test_img = X_train[ii,jj,:]
#         test_img = test_img.reshape(60,80)
#         cv2.imshow('test',test_img)
#         print "jj=", jj, ";", "ii=", ii, "\n"
#         cv2.waitKey()

this_opt = RMSprop()



X_train = X_train.reshape(data_import.shape[0],data_import.shape[1],data_import.shape[2]*data_import.shape[3])
test_x = test_x.reshape(test_x.shape[0],test_x.shape[1],test_x.shape[2]*test_x.shape[3])

# This section is using fixed extracted feature to train lstm
# cv2.namedWindow('test')
# cv2.namedWindow('test1')
feature_data = np.zeros(X_train.shape)
feature_tmp = np.zeros((X_train.shape[0]*X_train.shape[1],X_train.shape[2]))


indx=0
for jj in range(X_train.shape[0]):
    for ii in range(X_train.shape[1]):
        feature_tmp[indx,:] = X_train[jj,ii,:]
        indx += 1

result_feature = model_feature_extract.predict(feature_tmp)
feature_data = np.zeros((X_train.shape[0],X_train.shape[1],num_hidden))
indx=0
for jj in range(X_train.shape[0]):
    for ii in range(X_train.shape[1]):
        ttttt = X_train[jj,ii,:].max()
        if X_train[jj,ii,:].max() != 0 and X_train[jj,ii,:].min()!=0:
            feature_data[jj,ii,:] = result_feature[indx,:]
        indx += 1



















model1 = Sequential([Masking(0,input_shape = (X_train.shape[1],80*60))])
model1.add(TimeDistributedDense(num_hidden,'glorot_uniform','sigmoid',auto_encoder_weight,
                                   W_regularizer=l2(0.00),activity_regularizer=activity_l1(0.0)))

model1.add(LSTM(64,return_sequences=True))
model1.add(LSTM(128,return_sequences=True))
model1.add(Dropout(0.2))
model1.add(LSTM(32))
# model1.add(Dropout(0.2))
model1.add(Dense(26))
model1.add(Activation('softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='adam')


model1.fit(X_train, data_label, batch_size=batch_size, nb_epoch=nb_epoch,
                     show_accuracy=True, verbose=1, validation_data=(test_x,test_y))

