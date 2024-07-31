import tensorflow as tf

class FSRCNN(tf.keras.Model):
    """
    Build the network of FSRCNN (see https://arxiv.org/abs/1608.00367)

    Attributes:
        img_shape: input image shape (width, height, channels)
        scale: image resize factor
    """

    def __init__(self, img_shape, scale):
        """
        Args:
            img_shape: input image shape (width, height, channels)
            scale: image resize factor
        """

        # initialize the super-class
        super(FSRCNN, self).__init__(name='fsrcnn')
        # set attributes
        self.img_shape = img_shape
        self.scale = scale
        # build a network
        self.__build__()

    def __build__(self, d=56, s=12, m=4):
        """
        Build FSRCNN model

        Args:
            d: number of filters in first and last layers
            s: number of filters in middle layers
            m: number of layers with 3x3 filters
        """

        # input low-resolution image
        inputs = tf.keras.Input(shape=self.img_shape)

        # set filters and kernels
        filters = [d, s] + [s] * m + [d]
        kernels = [5, 1] + [3] * m + [1]
        # add relu convolutional layers
        x = inputs
        for filter, kernel in zip(filters, kernels):
            x = tf.keras.layers.Conv2D(filter, kernel, padding='same',
                kernel_initializer='he_normal', activation='relu')(x)
        # add a linear convolutional layer
        outputs = tf.keras.layers.Conv2DTranspose(self.img_shape[-1], 9,
            strides=self.scale, padding='same', kernel_initializer='he_normal')(x)

        # setup the network
        super().__init__(inputs=inputs, outputs=outputs)
