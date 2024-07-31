import tensorflow as tf
from .layer import Resize

class SRCNN(tf.keras.Model):
    """
    Build the network of SRCNN (see https://arxiv.org/abs/1501.00092)

    Attributes:
        img_shape: input image shape (width, height, channels)
        scale: image resize factor
        params: read yaml(default is None)
    """

    def __init__(self, img_shape, scale, params=None):
        """
        Args:
            img_shape: input image shape (width, height, channels)
            scale: image resize factor
            params: read yaml(default is None)
        """

        # initialize the super-class
        super(SRCNN, self).__init__(name='srcnn')
        # set attributes
        self.img_shape = img_shape
        self.scale = scale
        self.params = params
        # build a network
        if self.params is not None:
            if hasattr(self.params,'srcnn'):
                self.__build__(filters=self.params.srcnn['filters'], kernels=self.params.srcnn['kernels'])
                return
        self.__build__()

    def __build__(self, filters=[32, 16], kernels=[9, 1, 5]):
        """
        Build SRCNN model

        Args:
            filters: number of channels with last layer removed (last layer is the same as input channel)
            kernels: kernel size for each filter
        """
        NO_PADDING=False
        print('build srcnn: ', filters, kernels)

        # input low-resolution image
        inputs = tf.keras.Input(shape=self.img_shape)
        # resize an image
        if NO_PADDING:
            margin = sum([ (v-1) for v in kernels ] )
            x = Resize(self.scale, margin=int(margin))(inputs)
        else:
            x = Resize(self.scale)(inputs)
        # add relu convolutional layers
        if NO_PADDING:
            padding = 'valid'
        else:
            padding = 'same'
        for filter, kernel in zip(filters, kernels):
            x = tf.keras.layers.Conv2D(filter, kernel, padding=padding,
                kernel_initializer='he_normal', activation='relu')(x)
        # add a linear convolutional layer
        outputs = tf.keras.layers.Conv2D(self.img_shape[-1], kernels[-1],
            padding=padding, kernel_initializer='he_normal')(x)

        # setup the network
        super().__init__(inputs=inputs, outputs=outputs)
