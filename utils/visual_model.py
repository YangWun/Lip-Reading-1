from MultiImage import ImageTemp
import cv2 as cv
import theano

def visual_weight(model, weight_path, show_size=(60, 60), cols = 8):
    '''
    visualize weight/filters of the first layer.
    if filters has multiple channels, every channel will be show in different windows
    :param model: model to visualize
    :param weight_path: weight file path
    :param show_size: (width, height) filter size to show
    :param cols: how many filter images in one row
    :return: visual image
    '''

    # load weight
    model.load_weights(weight_path)
    weight = model.get_weights()
    filter_num = weight[0].shape[0]
    cha = weight[0].shape[1]
    rows = filter_num/cols
    if rows*cols < filter_num:
        rows += 1

    visual_img = []
    for i in range(cha):
        visual_img.append(ImageTemp(subimage=show_size, size=(rows, cols)))

    for c in range(cha):
        for i in range(filter_num):
            filter = weight[0][i, c, :, :]
            visual_img[c].fill(filter, place=(i//cols, i % cols))
        visual_img[c].show('weight-channel'+str(c))
    cv.waitKey()
    cv.destroyAllWindows()
    return visual_img

def visual_output(model, weight_path, layer_num, X, show_size=(80, 60), cols = 8):
    '''
    visual the output of the 'layer_num'th layer.
    if output has multiple channels, every channel will be show in different windows.
    :param model: model to visualize
    :param weight_path: weight file path
    :param layer_num: number of layer to visualize
    :param X: input samples
    :param show_size: (width, height) output image size to show
    :param cols: how many images to place in one row
    :return: visual image
    '''
    # load weight
    model.load_weights(weight_path)
    get_activations = theano.function([model.layers[0].input], model.layers[layer_num].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X)

    img_num = activations.shape[0]
    cha = activations.shape[1]
    rows = img_num/cols
    if rows*cols < img_num:
        rows += 1

    visual_img = []
    for i in range(cha):
        visual_img.append(ImageTemp(subimage=show_size, size=(rows, cols)))

    for c in range(cha):
        for i in range(img_num):
            filter = activations[i, c, :, :]
            visual_img[c].fill(filter, place=(i//cols, i % cols))
        visual_img[c].show('activations-channel'+str(c))
    cv.waitKey()
    cv.destroyAllWindows()
    return visual_img