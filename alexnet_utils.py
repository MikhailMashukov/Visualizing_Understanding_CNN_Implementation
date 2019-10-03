import numpy as np
# from imageio import imread, imresize    # For scipy >= 1.2, but imresize needs additional searching
from scipy.misc import imread, imresize

def imresize110(arr, size, interp='bilinear', mode=None):   # From scipy 1.1, to reimplement with scipy >= 1.2
    """ .. warning::

        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).

    Parameters
    ----------
    arr : ndarray
        The array of image to be resized.
    size : int, float or tuple
        * int   - Percentage of current size.
        * float - Fraction of current size.
        * tuple - Size of the output image (height, width).

    interp : str, optional
        Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
        'bicubic' or 'cubic').
    mode : str, optional
        The PIL image mode ('P', 'L', etc.) to convert `arr` before resizing.
        If ``mode=None`` (the default), 2-D images will be treated like
        ``mode='L'``, i.e. casting to long integer.  For 3-D and 4-D arrays,
        `mode` will be set to ``'RGB'`` and ``'RGBA'`` respectively.

    Returns
    -------
    imresize : ndarray
        The resized array of image.

    See Also
    --------
    toimage : Implicitly used to convert `arr` according to `mode`.
    scipy.ndimage.zoom : More generic implementation that does not use PIL.

    """
    im = toimage(arr, mode=mode)
    ts = type(size)
    if issubdtype(ts, numpy.signedinteger):
        percent = size / 100.0
        size = tuple((array(im.size)*percent).astype(int))
    elif issubdtype(type(size), numpy.floating):
        size = tuple((array(im.size)*size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return fromimage(imnew)


def preprocess_image_batch(image_paths, img_size=(256, 256), crop_size=(227, 227), color_mode="rgb", out=None):
    """
    Resize, crop and normalize colors of images 
    to make them suitable for alexnet_model (if default parameter values are chosen)
    
    This function is also from 
    https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py
    with only some minor changes
    """

    # Make function callable with single image instead of list
    if type(image_paths) is str:
        image_paths = [image_paths]

    img_list = []
    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        img = imresize(img, img_size)

        img = img.astype('float32')
        # Normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # We permute the colors to get them in the BGR order
        if color_mode == "bgr":
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
        img = img.transpose((2, 0, 1))

        if crop_size:
            img = img[:, (img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2
            , (img_size[1] - crop_size[1]) // 2:(img_size[1] + crop_size[1]) // 2]

        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images'
                         ' in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch

# def keras_set_image_data_format
# # K.set_image_dim_ordering('th')
#     K.set_image_data_format('channels_first')