from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, merge, Concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD
from keras import backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils import plot_model

from alexnet_utils import preprocess_image_batch

from alexnet_additional_layers import split_tensor, cross_channel_normalization
from decode_predictions import decode_classnames_json, decode_classnumber


def alexnet_model(weights_path=None):
    """
    Returns a keras model for AlexNet, achieving roughly 80% at ImageNet2012 validation set
    
    Model and weights from
    https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py
    and only slightly modified to work with TF backend
    """

    # K.set_image_dim_ordering('th')
    K.set_image_data_format('channels_first')
    inputs = Input(shape=(3, 227, 227))

    conv_1 = Conv2D(96, 11, strides=4, activation='relu', name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
                # 11  4  4 = 19, stride 8
    conv_2 = cross_channel_normalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    new_convs =[Conv2D(128, 5, activation="relu", name='conv_2_' + str(i + 1))
        (split_tensor(ratio_split=2, id_split=i)(conv_2)
         ) for i in range(2)]
                # 2 * 8 <- (shifted)    19 8 8 8 8 = 51
    conv_2 =  Concatenate(axis=1, name="conv_2")(new_convs)
    # conv_2 = merge([ \
    #     Conv2D(128, 5, activation="relu", name='conv_2_' + str(i + 1))
    #     (split_tensor(ratio_split=2, id_split=i)(conv_2)
    #      ) for i in range(2)])  # , mode='concat', concat_axis=1, name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
                # 51 8 8 = 67, stride 16
    conv_3 = cross_channel_normalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, 3, activation='relu', name='conv_3')(conv_3)
                # 2 * 8 + 16 <-    67 16 16

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = Concatenate(axis=1, name="conv_4")([
        Conv2D(192, 3, activation="relu", name='conv_4_' + str(i + 1))(
            split_tensor(ratio_split=2, id_split=i)(conv_4)
        ) for i in range(2)])   # , mode='concat', concat_axis=1, name="conv_4")

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = Concatenate(axis=1, name="conv_5")([
        Conv2D(128, 3, activation="relu", name='conv_5_' + str(i + 1))(
            split_tensor(ratio_split=2, id_split=i)(conv_5)
        ) for i in range(2)])   # mode='concat', concat_axis=1, name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, name='dense_3')(dense_3)
    prediction = Activation("softmax", name="softmax")(dense_3)

    m = Model(input=inputs, output=prediction)

    if weights_path is None:
        weights_path = 'Data/alexnet_weights.h5'
    m.load_weights(weights_path)
    # Model was trained using Theano backend
    # This changes convolutional kernels from TF to TH, great accuracy improvement
    convert_all_kernels_in_model(m)

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # m.compile(optimizer=sgd, loss='mse')

    return m


class AlexNet():
    """
    Wrapper for alexnet_model, makes calculating features of intermediate layers a one-liner
    Call with alexnet_model, if one already exists; otherwise one will be created
    If highest layer is given, predictions() will return output of convolution at that layer
    If not, predictions() will return 1000-dim vector of hot-encoded class probabilities
    """

    val_set_size = 50000
    channels = [3, 96, 256, 384, 384, 256]  # Corresponds to the number of filters in convolution (except first entry)
    conv_layer_names = ['conv_' + id for id in ('1', '2_1', '2_2', '3', '4_1', '4_2', '5_1', '5_2')]
    deconv_layer_names = ['deconv_' + id for id in ('1', '2_1', '2_2', '3', '4_1', '4_2', '5_1', '5_2')]

    def __init__(self, highest_layer=None, base_model=None):
        self.highest_layer = highest_layer
        # if base_model is None:
        #     print("None")
        self.base_model = base_model if base_model else alexnet_model()  # If no base_model, create alexnet_model
        self.model = self._sub_model() if highest_layer else self.base_model  # Use full network if no highest_layer

    def _sub_model(self):
        if isinstance(self.highest_layer, int):
            highest_layer_name = 'conv_{}'.format(self.highest_layer)
        else:
            highest_layer_name = self.highest_layer
            # if highest_layer_name[-len('_weights') : ] == '_weights':
            #     highest_layer = self.base_model.get_layer(highest_layer_name[ : -len('_weights')])._trainable_weights
            # else:
        highest_layer = self.base_model.get_layer(highest_layer_name)
        return Model(inputs=self.base_model.input,
                     outputs=highest_layer.output)

    def predict(self, img_path):
        """
        Takes the image path as argument, unlike alexnet_model.predict which takes the preprocessed array
        """
        img = preprocess_image_batch(img_path)
        return self.model.predict(img)

    def top_classes(self, img_path, top=5):
        preds = self.predict(img_path)
        return decode_classnumber(preds, top)


    # # Returns (multiplier, size). Source pixels, corresponding to the layer layerName's pixel (x, y)
    # # has coordinates (x * multiplier, y * multiplier) - (... + size - 1, ... + size - 1)
    # @staticmethod
    # def get_layer_source_pixel_calc_params(layerName):
    #     mult = 4
    #     size = 11
    #     if layerName == 'conv_1':
    #         return (mult, size)
    #     mult *= 2
    #     size += mult + mult * 4
    #     if layerName == 'conv_2':
    #         return (mult, size)
    #     mult *= 2
    #     size += mult + mult * 2
    #     if layerName == 'conv_3':
    #         return (mult, size)
    #     return (None, None)

    @staticmethod
    def get_source_block_calc_func(layerName):
        if layerName == 'conv_1':
            return AlexNet.get_conv_1_source_block
        elif layerName == 'conv_2':
            return AlexNet.get_conv_2_source_block
        elif layerName == 'conv_3':
            return AlexNet.get_conv_3_source_block
        elif layerName == 'conv_4':
            return AlexNet.get_conv_4_source_block
        elif layerName == 'conv_5':
            return AlexNet.get_conv_5_source_block
        elif layerName[:6] == 'dense_':
            return AlexNet.get_entire_image_block
        else:
            return None

    # Returns source pixels block, corresponding to the layer conv_1 pixel (x, y)
    @staticmethod
    def get_conv_1_source_block(x, y):
        source_xy_0 = (x * 4, y * 4)
        size = 11
        return (source_xy_0[0], source_xy_0[1], source_xy_0[0] + size, source_xy_0[1] + size)

    @staticmethod
    def get_conv_2_source_block(x, y):
        source_xy_0 = ((x - 2) * 8, (y - 2) * 8)
        size = 51  # 11 + 4 * 2 + 8 * 4
        return (0 if source_xy_0[0] < 0 else source_xy_0[0],
                0 if source_xy_0[1] < 0 else source_xy_0[1],
                source_xy_0[0] + size, source_xy_0[1] + size)

    @staticmethod
    def get_conv_3_source_block(x, y):
        source_xy_0 = ((x - 2) * 16, (y - 2) * 16)
        size = 99  # 51 + 8 * 2 + 16 * 2
        return (0 if source_xy_0[0] < 0 else source_xy_0[0],
                0 if source_xy_0[1] < 0 else source_xy_0[1],
                source_xy_0[0] + size, source_xy_0[1] + size)

    @staticmethod
    def get_conv_4_source_block(x, y):
        source_xy_0 = ((x - 3) * 16, (y - 3) * 16)
        size = 131  # 99 + 16 * 2
        return (0 if source_xy_0[0] < 0 else source_xy_0[0],
                0 if source_xy_0[1] < 0 else source_xy_0[1],
                source_xy_0[0] + size, source_xy_0[1] + size)

    @staticmethod
    def get_conv_5_source_block(x, y):
        source_xy_0 = ((x - 4) * 16, (y - 4) * 16)
        size = 163  # 131 + 16 * 2
        return (0 if source_xy_0[0] < 0 else source_xy_0[0],
                0 if source_xy_0[1] < 0 else source_xy_0[1],
                source_xy_0[0] + size, source_xy_0[1] + size)

    @staticmethod
    def get_entire_image_block(x, y):
        return (0, 0, 227, 227)

if __name__ == "__main__":
    img_path = 'Example_JPG/Elephant.jpg'

    # Usage of alexnet_model
    im = preprocess_image_batch([img_path])
    model = alexnet_model()
    out_model = model.predict(im)

    # Usage of AlexNet()
    out_wrapper = AlexNet().predict(img_path)

    assert (out_model == out_wrapper).all()

    # Decode one-hot vector to most probable class names
    print(decode_classnames_json(out_wrapper))
    print(decode_classnumber(out_wrapper))

    # Plot and print information about model
    plot_and_print = True
    if plot_and_print:
        plot_model(model, to_file='alexnet_model.png', show_shapes=True)
        print(model.summary())

    testimages = ['Example_JPG/Elephant.jpg', 'Example_JPG/RoadBike.jpg', 'Example_JPG/Trump.jpg']
    model = alexnet_model()
    preds = AlexNet(base_model=model).top_classes(testimages)
    print(preds)
    for pred in preds:
        print(pred)
