if 1:
    from keras.layers.core import Lambda
    from keras import backend as K
else:
    from tensorflow.keras.layers import Lambda
    from tensorflow.keras import backend as K

"""
Both layers from
https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py
and only slightly modified to work with TF backend as suggested by GitHub user 
"""

def cross_channel_normalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    # This is the function used for cross channel normalization in the original Alexnet

    def f_channelsFirst(X):
        K.set_image_data_format('channels_first')
        b, ch, r, c = X.get_shape()
        half = n // 2
        square = K.square(X)

        # Rearrange to make spatial_2d_padding work, then re-rearrange
        square = K.permute_dimensions(square, (0, 2, 3, 1))
        extra_channels = K.spatial_2d_padding(square, ((0, 0), (half, half)))
        extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))

        scale = k
        ch = int(ch)
        for i in range(n):
            scale += alpha * extra_channels[:, i:i + ch, :, :]
        scale = scale ** beta
        return X / scale

    def f_channelsLast(X):
        b, r, c, ch = X.get_shape()
        half = n // 2
        square = K.square(X)

        # Rearrange to make spatial_2d_padding work, then re-rearrange
        # square = K.permute_dimensions(square, (0, 2, 3, 1))
        square = K.permute_dimensions(square, (0, 1, 3, 2))
        extra_channels = K.spatial_2d_padding(square, ((0, 0), (half, half)), data_format=K.image_data_format())
            # Incorrectly pads to x (dimension 2 counting from 0) instead of channels
            # "spatial_2d_padding Pads the 2nd and 3rd dimensions of a 4D tensor."
            # https://keras.io/backend/
            # So we temporarily place channels to dimenstion 2
        extra_channels = K.permute_dimensions(extra_channels, (0, 1, 3, 2))
        # extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))

        scale = k
        ch = int(ch)
        for i in range(n):
            scale += alpha * extra_channels[:, :, :, i:i + ch]
        scale = scale ** beta
        return X / scale

    return Lambda(f_channelsFirst if K.image_data_format() == 'channels_first' else f_channelsLast,
                  output_shape=lambda input_shape: input_shape, **kwargs)

def split_tensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        div = int(X.get_shape()[axis]) // ratio_split

        if axis == 0:
            output = X[id_split * div:(id_split + 1) * div, :, :, :]
        elif axis == 1:
            output = X[:, id_split * div:(id_split + 1) * div, :, :]
        elif axis == 2:
            output = X[:, :, id_split * div:(id_split + 1) * div, :]
        elif axis == 3:
            output = X[:, :, :, id_split * div:(id_split + 1) * div]
        else:
            raise ValueError("This axis is not possible")

        return output

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)
