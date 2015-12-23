def test_lstm_auto_encoder():
    (data_import,data_label),(test_x,test_y) = read_data_sequence('./avletters/Lips/')
    #assign 'vid' element in "dict" structure to data



    batch_size = 128
    nb_epoch = 1000




    X_train = data_import
    X_train = X_train.astype('float32')


    ori_size = data_import.shape
    reshaped_xtrain = X_train.reshape(ori_size[0],ori_size[1],ori_size[2]*ori_size[3])
    new_size = reshaped_xtrain.shape
    re_test_x = test_x.reshape(test_x.shape[0],test_x.shape[1],test_x.shape[2]*test_x.shape[3])

    num_hidden = 64
    lstm_output_size = 20



    # for test_i in range(1000):
    #     first_img = X_train[test_i,:,:]
    #     cv2.imshow('111',first_img)
    #     cv2.waitKey()


    model_read = Sequential()


    this_opt = RMSprop()

    model_read.add(Dense(num_hidden,'glorot_uniform','sigmoid',None,l2(0.00),None,activity_l1(0.01),None,None,80*60))
    model_read.add(Dense(80*60,'glorot_uniform','linear',None,l2(0.00),None,None,None,None))
    model_read.load_weights('./weight_file')


    # cv2.namedWindow('test')
    # for ii in range(reshaped_xtrain.shape[0]):
    #     for jj in range(reshaped_xtrain.shape[1]):
    #         test_img = reshaped_xtrain[ii,jj,:]
    #         test_img = test_img.reshape(60,80)
    #         cv2.imshow('test',test_img)
    #         cv2.waitKey()


    auto_encoder_weight = model_read.layers[0].get_weights()
 #   t1 = auto_encoder_weight[0].transpose()
 #   auto_encoder_weight[0] = t1.reshape(num_hidden,ori_size[2]*ori_size[3],1,1)



    # # encoder = containers.Sequential([TimeDistributedDense(128, input_shape = (ori_size[1],80*60))])
    # # decoder = containers.Sequential([TimeDistributedDense(80*60, input_shape = (ori_size[1],128))])
    #
    # model.add(TimeDistributedDense(num_hidden,'glorot_uniform','sigmoid',None,
    #                                W_regularizer=l2(0.00),activity_regularizer=activity_l1(0.0)))










    # model.add(Masking(0,input_shape = (ori_size[1],80*60)))
    # use convolution layer to handle time dense data
    # model.add(Convolution1D(num_hidden,1,'uniform','sigmoid',auto_encoder_weight,border_mode='valid',
    #                     subsample_length=1,W_regularizer=l2(0.00),activity_regularizer=activity_l1(0.01),input_dim=80*60))





    (data_import,data_label),(test_x,test_y) = read_data('./avletters/Lips/')
    data_import = data_import.reshape(data_import.shape[0],80*60)


    # model.add(TimeDistributedDense(num_hidden,'glorot_uniform','sigmoid',auto_encoder_weight,
    #                                W_regularizer=l2(0.00),activity_regularizer=activity_l1(0.0)))
    # model.add(Masking(0))
    # model.add(LSTM(lstm_output_size))
    model1 = Sequential([model_read.layers[0]])
    # model.add(Dense(64,weights=auto_encoder_weight,activation='sigmoid',input_dim=60*80))
    model1.add(Dense(26))
    model1.add(Activation('softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer=this_opt)


    model1.fit(data_import, data_label, batch_size=batch_size, nb_epoch=nb_epoch,
               show_accuracy=True, verbose=1, validation_data=(data_import,data_label))


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















ttt = feature_data[jj,ii,:].reshape(60,80)
ttt1 =  X_train[jj,ii,:].reshape(60,80)
cv2.imshow('test',ttt)
cv2.imshow('test1',ttt1)
print "jj=", jj, "ii=", ii
cv2.waitKey()