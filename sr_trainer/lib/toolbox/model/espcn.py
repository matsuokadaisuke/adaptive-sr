import tensorflow as tf
from .layer import Conv2DSubPixel

class ESPCN(tf.keras.Model):
    """
    Build the network of ESPCN (see https://arxiv.org/abs/1609.05158)

    Attributes:
        img_shape: input image shape (width, height, channels)
        scale: image resize factor
    """

    def __init__(self, img_shape, scale, params=None):
        """
        Args:
            img_shape: input image shape
            scale: image resize factor
        """

        # initialize the super-class
        # super(ESPCN, self).__init__(name='espcn')
        # set attributes
        self.img_shape = img_shape
        self.scale = scale
        self.params = params
        self.filters = [64,32]
        self.kernels = [5,3,3]
        if self.params is not None:
            if hasattr(self.params, 'espcn'):
                self.filters = self.params.espcn['filters']
                self.kernels = self.params.espcn['kernels']
        # build a network
        self.__build__(filters=self.filters, kernels=self.kernels)

    def __build__(self, filters=[64, 32], kernels=[5, 3, 3]):
        """
        Build ESPCN model

        Args:
            filters: number of channels with last layer removed (last layer is the same as input channel)
            kernels: kernel size for each filter
        """
        
        print(filters, kernels)
        # input low-resolution image
        inputs = tf.keras.Input(shape=self.img_shape)

        # add relu convolutional layers
        x = inputs
        for filter, kernel in zip(filters, kernels):
            x = tf.keras.layers.Conv2D(filter, kernel, padding='same',
                kernel_initializer='he_normal', activation='relu')(x)
        # add a linear convolutional layer
        x = tf.keras.layers.Conv2D(self.img_shape[-1] * self.scale ** 2,
            kernels[-1], padding='same', kernel_initializer='he_normal')(x)
        # add sub-pixel convolution layer
        outputs = Conv2DSubPixel(self.scale)(x)

        # setup the network
        super().__init__(inputs=inputs, outputs=outputs)
