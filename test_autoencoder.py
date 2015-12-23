from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

#import layers
from keras.datasets import mnist
from keras.optimizers import RMSprop


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, AutoEncoder,Permute,Reshape, Masking,TimeDistributedDense
from keras.layers.core import ActivityRegularization
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Convolution1D
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.layers import containers
from keras.regularizers import l2, activity_l1
from keras.layers.convolutional import Convolution1D
from core_processing import read_data_motion

#import visualization lib
from keras.utils.visualize_util import plot
import theano



import os
import os.path
import scipy.io as matio
import cv2
from read_data import read_data
from read_data import read_data_sequence

def test_auto_encoder():
    (data_import,data_label),(test_x,test_y) = read_data_motion('./avletters/Lips/')
    #assign 'vid' element in "dict" structure to data



    batch_size = 128
    nb_epoch = 1000

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



    Y_train = X_train
    Y_test = X_test

    num_sample = X_train.shape[0]
    num_dim = X_train.shape[1]*X_train.shape[2]
    num_hidden = 64

    #X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    #X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

    X_train = X_train.reshape(X_train.shape[0],X_train.shape[2],X_train.shape[3])

    #X_train = X_train[range(100),:,:]
    cv2.namedWindow('111')


    # for test_i in range(1000):
    #     first_img = X_train[test_i,:,:]
    #     cv2.imshow('111',first_img)
    #     cv2.waitKey()


    model = Sequential()





    model.add(Dense(num_hidden,'glorot_uniform','sigmoid',None,l2(0.00),None,activity_l1(0.01),None,None,80*60))
    #model.add(Dense(48,'glorot_uniform','sigmoid',None,l2(0.00),None,None,None,None))
    model.add(Dense(80*60,'glorot_uniform','linear',None,l2(0.00),None,None,None,None))


    model.compile(loss='mean_squared_error', optimizer='adadelta')



    # encoder = containers.Sequential([Dense(num_sample, input_dim=num_dim), Dense(num_hidden,'glorot_uniform','linear',None,l2(0.01),None,None,None,None)])
    # decoder = containers.Sequential([Dense(num_sample, input_dim=num_hidden), Dense(num_dim,'glorot_uniform','linear',None,l2(0.01),None,None,None,None)])
    # model.add(AutoEncoder(encoder=encoder, decoder=decoder,output_reconstruction=True))
    # model.add(AutoEncoder(encoder=encoder, decoder=decoder,output_reconstruction=True))


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
        min_v = min(out)
        max_v = max(out)
        #out = out - min_v
        #out = out/(max_v-min_v)
        out = out.reshape(60,80)
        cv2.imshow('111',out)
        cv2.waitKey()







# lstm test
def test_lstm_auto_encoder():
    (data_import,data_label),(test_x,test_y) = read_data('../avletters/Lips/')
    #assign 'vid' element in "dict" structure to data




    batch_size = 128
    nb_epoch = 1000




    X_train = data_import
    X_train = X_train.astype('float32')



    num_hidden = 64



    model = Sequential()




    this_opt = RMSprop()

    model.add(Dense(num_hidden,'glorot_uniform','sigmoid',None,l2(0.00),None,activity_l1(0.0),None,None,80*60))
    model.add(Dense(80*60,'glorot_uniform','linear',None,l2(0.00),None,None,None,None))
    model.compile(loss='mean_squared_error', optimizer='adadelta')
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[2]*X_train.shape[3])
    Y_train = X_train
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,
              show_accuracy=True, verbose=1, validation_data=(X_train, Y_train))



    model1 = Sequential([model.layers[0]])
    # model.add(Dense(64,weights=auto_encoder_weight,activation='sigmoid',input_dim=60*80))
    model1.add(Dense(26))
    model1.add(Activation('softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer=this_opt)


    model1.fit(X_train, data_label, batch_size=batch_size, nb_epoch=nb_epoch,
               show_accuracy=True, verbose=1, validation_data=(X_train,data_label))



















