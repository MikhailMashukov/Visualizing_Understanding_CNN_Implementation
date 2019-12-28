# Part of ImageNetsVisWrappers.py, extracted to make lazy import tensorflow properly
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import datetime
import math
import numpy as np

import ImageModels
from MnistModel2 import get_tensor_array_element
from MyUtils import getCpuCoreCount

# Net based on model, closer by style to alexnet
class CImageRecognitionNet:
    def __init__(self, highest_layer=None, base_model=None):
        self.highest_layer = highest_layer
        # if base_model is None:
        #     print("None")
        self.base_model = base_model
        self.batchSize = 32
        self.createModel()
        # super(CMnistRecognitionNet2, self).__init__()

    def init(self, imageDataset, logDir):
        self.imageDataset = imageDataset
        self.logDir = logDir

        # self.train_ds = tf.data.Dataset.from_tensor_slices(
        #         self.mnistDataset.getNetSource('train')).shuffle(5000).batch(32)
        # self.test_ds = tf.data.Dataset.from_tensor_slices(
        #         self.mnistDataset.getNetSource('test')).batch(32)

        # self.train_writer = tf.compat.v1.summary.FileWriter(self.logDir + '/train', sess.graph)
        # self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logDir, histogram_freq=1)
        self.train_writer = tf.summary.create_file_writer(self.logDir + '/train')
        self.test_writer  = tf.summary.create_file_writer(self.logDir + '/test')
        # self.watcher = tw.Watcher(filename=self.logDir + '/watch.log')
        # self.watch_stream = self.watcher.create_stream(name='metric1')
        # self.watcher.make_notebook()

        # self.createModel()

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        # self.gradTfVar = tf.Variable(np.zeros([128, 10], dtype=np.float32), name='grad_var')

        self.trainIterNum = 0

    def createModel(self):
        if not self.base_model:
            # self.base_model = MnistModel2.CMnistModel2()   # If no base_model, create net
            if 1:
                self.base_model = ImageModels.CImageModel()
            else:
                # import alexnet

                # self.base_model = alexnet.alexnet_model(None)
                self.base_model = ImageModels.MyAlexnetModel()
        self.model = self._sub_model() if self.highest_layer else self.base_model         # Use full network if no highest_layer

    def _sub_model(self):
        from keras.models import Model

        if isinstance(self.highest_layer, int):
            highest_layer_name = 'conv_{}'.format(self.highest_layer)
        else:
            highest_layer_name = self.highest_layer
            # if highest_layer_name[-len('_weights') : ] == '_weights':
            #     highest_layer = self.base_model.get_layer(highest_layer_name[ : -len('_weights')])._trainable_weights
            # else:
        try:
            highest_layer = self.base_model.get_layer(highest_layer_name)
            return Model(inputs=self.base_model.input,
                         outputs=highest_layer.output)
        except ValueError as valueEx:
            highest_layer_output = self.base_model.debug_layers[highest_layer_name]
            return Model(inputs=self.base_model.input,
                         outputs=highest_layer_output)

    @staticmethod
    def _transformLabelsForNet(labels):
        return tf.keras.utils.to_categorical(labels, num_classes=10)

    # # Pair - (imageNum, label)
    # def _loadTrainImage(self, pair):
    #     imageData = self.imageDataset.getImage(pair[0], 'net', 'train')
    #     return (imageData, pair[1])

    # def _loadTrainImage(self, imageNum):
    #     imageData = self.imageDataset.getImage(imageNum, 'net', 'train')
    #     return (imageData, self.imageDataset.getImageLabel(imageNum, 'train'))
    #
    # # @staticmethod
    # def _tfLoadTrainImage(imageNum):
    #     [imageData, ] = tf.py_function(self._loadTrainImage, [imageNum], [tf.float32])

    # def tf_random_rotate_image(image, label):
    #     im_shape = image.shape
    #     [image,] = tf.py_function(random_rotate_image, [image], [tf.float32])
    #     image.set_shape(im_shape)
    #     return image, label

    def _getTfDatasets(self, trainImageNums, testImageNums,
                      epochImageCount):
        trainDatasetImageCount = trainImageNums.shape[0]
        classCount = self.imageDataset.getClassCount()

        if epochImageCount == trainDatasetImageCount:
            curTrainDataset = trainImageNums
                       # , self._transformLabelsForNet(trainDataset[1]))
        else:
            # assert not 'Not implemented'
            permut = np.random.permutation(trainDatasetImageCount)  \
                    [ : (epochImageCount // self.batchSize + 10) * self.batchSize]
            curTrainDataset = trainImageNums[permut]
            # curTrainDataset = (trainDataset[0][permut, :, :, :],
            #            self._transformLabelsForNet(trainDataset[1][permut]))

        # curTestDatasetSize = epochImageCount // 6
        # if curTestDatasetSize >= testDataset[0].shape[0]:
        #     curTestDataset = (testDataset[0],
        #            self._transformLabelsForNet(testDataset[1]))
        # else:
        #     assert not 'Not implemented'
        #     permut = np.random.permutation(testDataset[0].shape[0])[:curTestDatasetSize]
        #     curTestDataset = (testDataset[0][permut, :, :, :],
        #                    self._transformLabelsForNet(testDataset[1][permut]))
        # # curTestDataset = (testDataset[0][:curTestDatasetSize],
        # #            self._transformLabelsForNet(testDataset[1][:curTestDatasetSize]))

        if 0:
            def _loadTrainImage(imageNum):
                imageData = self.imageDataset.getImage(imageNum, 'net', 'train')
                return (imageData, tf.keras.utils.to_categorical(
                            np.array(self.imageDataset.getImageLabel(imageNum, 'train')), num_classes=classCount))

            def _tfLoadTrainImage(imageNum):
                x = tf.py_function(_loadTrainImage, [imageNum], [tf.float32, tf.int32])
                [image, label] = x
                return image, label

            # def _fixup_shape(images, labels):
            #     images.set_shape([None, 227, 227, 3])
            #     labels.set_shape([None, classCount])
            #     # weights.set_shape([None])
            #     return images, labels
            # ds = tfds.load('dataset', split='train',   as_supervised=True)

            tfTrainDataset = tf.data.Dataset.from_tensor_slices(curTrainDataset)
            tfTrainDataset = tfTrainDataset.shuffle(epochImageCount * 2).map(_tfLoadTrainImage)
            tfTrainDataset = tfTrainDataset.batch(self.batchSize).prefetch(2)  # .map(_fixup_shape)
                # TODO: tfTrainDataset object caching
            # for images, labels in tfTrainDataset.take(2):
            #     print('My dataset labels shape', labels.numpy().shape)

        if 0:
            imgGen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)
            flowFromDirParams = dict(directory=self.imageDataset.getImageFolder(), \
                                     target_size=(227, 227), class_mode='categorical', seed=1, \
                                     batch_size=self.batchSize) # , save_to_dir='Data/Augmented')
            # flowFromDirParams = {'directory': self.imageDataset.getImageFolder(), \
            #                     'target_size': (227, 227), 'class_mode': 'categorical', 'seed': 1, \
            #                     'save_to_dir'': 'Data/Augmented'}
                # https://keras.io/preprocessing/image/, the flow_from_directory section
                # The dictionary containing the mapping from class names to class indices can be obtained via the attribute class_indices.
            imgGenFlow = imgGen.flow_from_directory(**flowFromDirParams)
            # images, labels = next(imgGen)
            # dataset.list_files

            def generator():
                return imgGen.flow_from_directory(**flowFromDirParams, batch_size=1)

            # tfTrainDataset = tf.data.Dataset.from_generator(generator, # args=flowFromDirParams,
            #             output_types=(tf.float32, tf.float32), output_shapes=([32,227,227,3], [32,classCount]))
            tfTrainDataset = tf.data.Dataset.from_generator(generator, # args=flowFromDirParams,
                        output_types=(tf.float32, tf.float32), output_shapes=([227,227,3], [classCount]))
            try:
                tfTrainDataset = tfTrainDataset.shuffle(max(epochImageCount, 4000))
                tfTrainDataset = tfTrainDataset.batch(self.batchSize)
                tfTrainDataset = tfTrainDataset.prefetch(2)
                # tfTrainDataset = tfTrainDataset.repeat(2)
                # tfTrainDataset = iter(tfTrainDataset)
                tfTrainDataset = tfTrainDataset.make_one_shot_iterator()
                next(tfTrainDataset)  # dir(tfTrainDataset)
                # next(tfTrainDataset.make_one_shot_iterator())
            except:
                pass


        if 0:
            import tensorflow_datasets as tfds

            tfCifarDataset = tfds.image.Cifar100(data_dir='Data/TfdsCifar')
            tfCifarDataset.download_and_prepare()
            for info in tfCifarDataset.as_dataset(split='train', as_supervised=True).take(4):
                # label = info['label'].numpy()
                label = info[1].numpy()
                # print('Labels: ', label)
            # tfTrainDataset = tfCifarDataset.as_dataset(split='train')
            # tfds.show_examples(tfCifarDataset.info, tfTrainDataset)

        # try:
        #     tfDataset = tfds.image.Imagenet2012()
        #     tfDataset.download_and_prepare(download_dir='Data/TfdsImageNet')
        #     print('Imagenet dataset downloaded')
        #
        #     # for images, labels in tfDataset.take(10):
        #     for images, labels in loaded:
        #         images.numpy()
        #         labels.numpy()
        #         print('Labels: ', labels)
        # except:
        #     pass

        if 1:
            from keras.preprocessing.image import ImageDataGenerator

            if 1:
                # CPU augmentation
                import alexnet_utils
                import random

                datagen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.1, rotation_range=30)
                    # With all 3 at first 16 mini-epochs and only horizontal_flip after - 89% accuracy on train
                    # and 69% on test
                img_size=(256, 256)
                crop_size=(227, 227)

                def prepareBatch(imageNums, labels):
                    images = []
                    rands = np.random.randint(1, high=img_size[0] - crop_size[0], size=(self.batchSize * 2))
                    randI = 0
                    for imageNum in imageNums:
                        image = self.imageDataset.getImage(imageNum, 'resized256', 'train').astype(np.float32)
                        image[:, :, 0] -= 123
                        image[:, :, 1] -= 116
                        image[:, :, 2] -= 103

                        image = next(datagen.flow(np.expand_dims(image, axis=0), batch_size=1))[0]
                        image = image[rands[randI] : rands[randI] + crop_size[0],
                                      rands[randI + 1] : rands[randI + 1] + crop_size[1], :]
                        images.append(image)
                        randI += 2
#                     images[:, :, :, 0] -= 123.68
#                     images[:, :, :, 1] -= 116.779
#                     images[:, :, :, 2] -= 103.939
#                     print(len(images), images[0].shape, images[0].dtype)

                    images = np.stack(images)
                    return images, tf.keras.utils.to_categorical(
                                labels, num_classes=classCount)

                def tfPrepareBatch(imageNums, labels):
                    images, labels = tf.py_function(prepareBatch, [imageNums, labels], [tf.int32, tf.int32])
                    return images, labels

                tfTrainDataset = self.imageDataset.getTfDataset('train')   # .take(100).repeat()
                tfTrainDataset = tfTrainDataset.shuffle(max(epochImageCount, 12000))
                tfTrainDataset = tfTrainDataset.batch(self.batchSize)
                tfTrainDataset = tfTrainDataset.map(tfPrepareBatch, num_parallel_calls=4)
                tfTrainDataset = tfTrainDataset.prefetch(3)
            else:
                # TODO: GPU augmentation
                # import alexnet_utils
                # import random

                # img_size=(256, 256)
                # crop_size=(227, 227)

                def prepareBatch(imageNums, labels):
                    images = []
                    # rands = np.random.randint(1, high=img_size[0] - crop_size[0], size=(self.batchSize * 2))
                    # randI = 0
                    for imageNum in imageNums:
                        image = self.imageDataset.getImage(imageNum, 'resized256', 'train').astype(np.float32)
                        # image[:, :, 0] -= 123
                        # image[:, :, 1] -= 116
                        # image[:, :, 2] -= 103
                        #
                        # image = next(datagen.flow(np.expand_dims(image, axis=0), batch_size=1))[0]
                        # image = image[rands[randI] : rands[randI] + crop_size[0],
                        #               rands[randI + 1] : rands[randI + 1] + crop_size[1], :]
                        images.append(image)
                        # randI += 2
#                     print(len(images), images[0].shape, images[0].dtype)

                    images = np.stack(images)
                    return images, tf.keras.utils.to_categorical(
                                labels, num_classes=classCount)

                def tfPrepareBatch(imageNums, labels):
                    images, labels = tf.py_function(prepareBatch, [imageNums, labels], [tf.int32, tf.int32])
                    return images, labels

                tfTrainDataset = self.imageDataset.getTfDataset('train')   # .take(100).repeat()
                tfTrainDataset = tfTrainDataset.shuffle(max(epochImageCount, 12000))
                tfTrainDataset = tfTrainDataset.batch(self.batchSize)
                tfTrainDataset = tfTrainDataset.map(tfPrepareBatch, num_parallel_calls=4)
                tfTrainDataset = tfTrainDataset.prefetch(3)

            # curTestDatasetSize = epochImageCount // 6
            # if curTestDatasetSize >= len(testImageNums):
            #     curTestDatasetSize = len(testImageNums)

            def _loadTestImage(imageNum, label):
                imageData = self.imageDataset.getImage(imageNum, 'net', 'test')
                return (imageData, tf.keras.utils.to_categorical(
                            label, num_classes=classCount))

            def _tfLoadTestImage(imageNum, label):
                x = tf.py_function(_loadTestImage, [imageNum, label], [tf.float32, tf.int32])
                # [image, label] = x
                return x

            tfTestDataset = self.imageDataset.getTfDataset('test')   # .take(100).repeat()
            tfTestDataset = tfTestDataset.shuffle(max(epochImageCount, 2000))
            # tfTestDataset = tfTestDataset.take(curTestDatasetSize)
            tfTestDataset = tfTestDataset.map(_tfLoadTestImage, num_parallel_calls=3)
            tfTestDataset = tfTestDataset.batch(self.batchSize)
            tfTestDataset = tfTestDataset.prefetch(2)


        if 0:
            model2 = tf.keras.models.Sequential([
              tf.keras.layers.Flatten(input_shape=(227, 227, 3)),
              tf.keras.layers.Dense(16, activation='relu'),
              tf.keras.layers.Dense(classCount, activation='softmax')
            ])
            model2.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            # model2.fit(tfTrainDataset, epochs=1)
            model2.fit_generator(tfTrainDataset, epochs=1,
                        steps_per_epoch=epochImageCount // self.batchSize)     # Error The shape of labels (800,), received (32, 25).

        # inp = [curTrainDataset[0], np.ones([curTrainDataset[0].shape[0], 4])]
        # valData = curTestDataset[0], np.ones([curTestDataset[0].shape[0], 4])], curTestDataset[1]]
        # history = self.model.fit(tfTrainDataset, # TODO validation_data=curTestDataset,
        #                          epochs=initialEpochNum + epochCount, initial_epoch=initialEpochNum,
        #                          # batch_size=self.batchSize,
        #                          verbose=2, callbacks=[tensorBoardCallback])
            # Exception 'PrefetchDataset' object is not an iterator

        return tfTrainDataset, tfTestDataset

    # Each dataset = (list/np.array of image numbers, np.array of labels (numbers in [0, number of classes)))
    def doLearning(self, epochCount, learningCallback,
                   trainImageNums, testImageNums,   # trainDataset, testDataset,
                   epochImageCount=None, initialEpochNum=0):
        from keras.callbacks import TensorBoard

        # epochCount = int(math.ceil(iterCount / 100))
        # trainDataset = self.mnistDataset.getNetSource('train')
        # testDataset = self.mnistDataset.getNetSource('test')
        trainDatasetImageCount = trainImageNums.shape[0]
        if epochImageCount is None or epochImageCount > trainDatasetImageCount:
            epochImageCount = trainDatasetImageCount
        # print("Running %d epoch(s) from %d, %d images each" % \
        #         (epochCount, initialEpochNum, epochImageCount))
        groupStartTime = datetime.datetime.now()

        tensorBoardCallback = TensorBoard(log_dir=self.logDir, histogram_freq=0,
                write_graph=False, write_grads=False, write_images=0,    # batch_size=32,
                # embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=True, embeddings_data=None,
                update_freq='epoch')
        # TODO summaryCallback = MnistModel2.CSummaryWriteCallback(self.imageDataset,
        #         self.train_writer, self.test_writer,
        #         int(initialEpochNum * trainImageNums.shape[0] / self.batchSize),
        #         learningCallback)
        # tensorBoardCallback.set_model(self.model)

        curTestDatasetSize = epochImageCount // 6
        if curTestDatasetSize >= len(testImageNums):
            curTestDatasetSize = len(testImageNums)
        tfTrainDataset, tfTestDataset = self._getTfDatasets(trainImageNums, testImageNums,
                epochImageCount)
        tfTrainDataset = tf.compat.v1.data.make_one_shot_iterator(tfTrainDataset)
        tfTestDataset = tf.compat.v1.data.make_one_shot_iterator(tfTestDataset)

        history = self.model.fit_generator(tfTrainDataset,
                                 epochs=initialEpochNum + epochCount, initial_epoch=initialEpochNum,
                                 steps_per_epoch=epochImageCount // self.batchSize,
                                 validation_data=tfTestDataset, validation_steps=curTestDatasetSize // self.batchSize,
                                 verbose=2) #, callbacks=[tensorBoardCallback])
            #, summaryCallback])
            # Without make_one_shot_iterator - error fused convolution not supported

        try:
            if not history.history:
                raise Exception('empty history object')
            infoStr = "loss %.5f" % history.history['loss'][-1]
            infoStr += ", acc %.4f" % history.history['accuracy'][-1]
        except Exception as ex:
            print("Error in doLearning: %s" % str(ex))
            infoStr = ''

        infoStr = "%s, last %d epochs: %.4f s" % \
                  (infoStr, epochCount,
                   (datetime.datetime.now() - groupStartTime).total_seconds())
                # groupStartTime = datetime.datetime.now()

        return infoStr


    def getImageLosses(self, startImageNum=1, imageCount=None):
        # from keras.callbacks import TensorBoard
        import keras.losses

        batchSize = min(getCpuCoreCount() * 64, 384)
        fullDataset = self.mnistDataset.getNetSource('train')
        fullDatasetImageCount = fullDataset[0].shape[0]
        if imageCount is None:
            imageCount = fullDatasetImageCount
        groupStartTime = datetime.datetime.now()

        data = (fullDataset[0][startImageNum - 1 : startImageNum + imageCount - 1, :, :, :],
                self._transformLabelsForNet(fullDataset[1][startImageNum - 1 : startImageNum + imageCount - 1]))
        # tfDataset = ...
        outputs = self.model.predict(data[0], verbose=1)   # batch_size=batchSize
        losses = keras.losses.mean_squared_error(outputs, data[1])
        return (losses, outputs)

        # losses = []
        # outputs = []
        # for batchNum in range((imageCount - 1) // batchSize + 1):
        #     imageInds = range(batchNum * batchSize + startImageNum - 1,
        #                       min((batchNum + 1) * batchSize, imageCount) + startImageNum - 1)
        #     batchData = (fullDataset[0][imageInds, :, :, :],
        #                  self._transformLabelsForNet(fullDataset[1][imageInds]))
        #     batchOutputs = self.model.predict(batchData[0], steps=1, verbose=1)
        #     batchLosses = keras.losses.mean_squared_error(batchOutputs, batchData[1])
        #     losses.append(batchLosses.numpy())
        #     outputs.append(batchOutputs)
        # return (np.concatenate(losses, axis=0), np.concatenate(outputs, axis=0))