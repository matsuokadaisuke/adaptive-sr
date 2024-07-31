import numpy as np
import tensorflow as tf
from .layer import Resize, Conv2DSubPixel

class Generator(tf.keras.Model):
    """
    Build the generator network of SRGAN (see https://arxiv.org/abs/1609.04802)

    Attributes:
        img_shape: input image shape (width, height, channels)
        scale: image resize factor
    """

    def __init__(self, img_shape, scale):
        """
        Args:
            img_shape: input low-resolution image shape
            scale: image resize factor
        """

        # initialize the super-class
        super(Generator, self).__init__(name='generator')
        # set attributes
        self.img_shape = img_shape
        self.scale = scale
        # build a network
        self.__build__()

    def __residual_block(self, filters, kernel_initializer, momentum=0.8):
        """
        Residual block with convolution and batch normalization

        Args:
            filters: number of kernels
            kernel_initializer: kernel initializer
            momentum: momentum for the moving mean and the moving variance

        Returns:
            residual block layer
        """
        def f(input):
            x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=kernel_initializer)(input)
            x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
            x = tf.keras.layers.PReLU(shared_axes=[1,2],
                alpha_initializer=kernel_initializer)(x)
            x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=kernel_initializer)(x)
            x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
            return tf.keras.layers.Add()([input, x])
        return f

    def __upsampling_block(self, filters, kernel_initializer):
        """
        Upsampling block with subpixel convolution

        Args:
            filters: base number of kernels
            kernel_initializer: kernel initializer

        Returns:
            upsampling block layer
        """
        def pixel_shuffle(x):
            """
            SRGAN default upsampling (pixel shuffle) layer
            """
            x = tf.keras.layers.Conv2D(4 * filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=kernel_initializer)(x)
            x = Conv2DSubPixel(scale=2)(x)
            return tf.keras.layers.PReLU(shared_axes=[1,2],
                alpha_initializer=kernel_initializer)(x)
                
        def resize_conv(x):
            """
            Resize-convolution layer, suggested by distill.pub
            to eliminate checkerboard artifacts
            (see https://distill.pub/2016/deconv-checkerboard/)
            """
            x = Resize(scale=2, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)(x)
            x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=kernel_initializer)(x)
            return tf.keras.layers.PReLU(shared_axes=[1,2],
                alpha_initializer=kernel_initializer)(x)
        return pixel_shuffle

    def __build__(self, filters=32, num_residual_blocks=1,
        kernel_initializer='he_normal', momentum=0.8):
        """
        Build the generator network of SRGAN

        Args:
            filters: base number of kernels
            num_residual_blocks: number of residual blocks
            kernel_initializer: kernel initializer
            momentum: momentum for the moving mean and the moving variance
        """

        # input low-resolution image
        inputs = tf.keras.Input(shape=self.img_shape)

        # pre-residual block
        x = tf.keras.layers.Conv2D(filters, kernel_size=9, strides=1, padding='same',
            kernel_initializer=kernel_initializer)(inputs)
        x = tf.keras.layers.PReLU(shared_axes=[1,2], alpha_initializer=kernel_initializer)(x)
        # residual blocks
        r = x
        for _ in range(num_residual_blocks):
            r = self.__residual_block(filters,
                kernel_initializer=kernel_initializer, momentum=momentum)(r)
        # post-residual block
        r = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
            kernel_initializer=kernel_initializer)(r)
        r = tf.keras.layers.BatchNormalization(momentum=momentum)(r)
        x = tf.keras.layers.Add()([x, r])
        # upsampling blocks
        num_upsampling_blocks = int(np.log2(self.scale))
        for _ in range(num_upsampling_blocks):
            x = self.__upsampling_block(
                filters, kernel_initializer=kernel_initializer)(x)

        # output high-resolution image
        outputs = tf.keras.layers.Conv2D(self.img_shape[-1],
            kernel_size=9, strides=1, padding='same',
            kernel_initializer=kernel_initializer)(x)

        # setup the network
        super().__init__(inputs=inputs, outputs=outputs, name='g')

class Discriminator(tf.keras.Model):
    """
    Build the discriminator network of SRGAN (see https://arxiv.org/abs/1609.04802)

    Attributes:
        img_shape: input image shape (width, height, channels)
        scale: image resize factor
    """

    def __init__(self, img_shape, scale):
        """
        Args:
            img_shape: input low-resolution image shape
            scale: image resize factor
        """

        # initialize the super-class
        super(Discriminator, self).__init__(name='discriminator')
        # set attributes
        self.img_shape = img_shape
        self.scale = scale
        # build a network
        self.__build__()

    def __conv2d_block(self, filters, kernel_size=3, strides=1,
        use_bn=True, kernel_initializer='he_normal', momentum=0.8, alpha=0.2):
        """
        Convolution block with batch normalization and leaky ReLU activation

        Args:
            filters: number of kernels
            kernel_size: height and width of the 2D convolution window
            strides: strides of the convolution along the height and width
            use_bn: If True, batch normalization is used. If False, one is not used.
            kernel_initializer: kernel initializer
            momentum: momentum for the moving mean and the moving variance
            alpha: LeakyReLU slope coefficient

        Returns:
            convolution block layer
        """
        def f(x):
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            strides=strides, padding='same')(x)
            if use_bn:
                x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
            return tf.keras.layers.LeakyReLU(alpha=alpha)(x)
        return f

    def __build__(self, filters=4, num_downsampling_blocks=1,
        kernel_initializer='he_normal', momentum=0.8, alpha=0.2):
        """
        Build the discriminator network of SRGAN

        Args:
            filters: base number of kernels
            num_downsampling_blocks: number of downsampling blocks
            kernel_initializer: kernel initializer
            momentum: momentum for the moving mean and the moving variance
            alpha: LeakyReLU slope coefficient
        """

        # get hig-resolution image shape
        hr_img_shape = (self.scale * self.img_shape[0],
                        self.scale * self.img_shape[1], self.img_shape[2])
        # build a network
        inputs = tf.keras.Input(shape=hr_img_shape)

        # pre-downsampling block
        x = self.__conv2d_block(filters, use_bn=False,
            kernel_initializer=kernel_initializer, momentum=momentum, alpha=alpha)(inputs)
        x = self.__conv2d_block(filters, strides=2,
            kernel_initializer=kernel_initializer, momentum=momentum, alpha=alpha)(x)
        # downsampling blocks
        for i in range(num_downsampling_blocks):
            num_kernels = 2**(i + 1) * filters
            x = self.__conv2d_block(num_kernels,
                kernel_initializer=kernel_initializer, momentum=momentum, alpha=alpha)(x)
            x = self.__conv2d_block(num_kernels, strides=2,
                kernel_initializer=kernel_initializer, momentum=momentum, alpha=alpha)(x)
        # post-downsampling blocks
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(4 * filters, kernel_initializer=kernel_initializer)(x)
        x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)

        # output the real or fake label
        outputs = tf.keras.layers.Dense(1, kernel_initializer=kernel_initializer,
            activation='sigmoid')(x)

        # setup the network
        super().__init__(inputs=inputs, outputs=outputs, name='d')

class SRGAN(tf.keras.Model):
    """
    Build the SRGAN network (see https://arxiv.org/abs/1609.04802)

    Attributes:
        img_shape: input image shape (width, height, channels)
        scale: image resize factor
        label_noise: noise of discriminator labels
        loss_weights: generator and discriminator loss weights
        generator: generator network
        discriminator: discriminator network
    """

    def __init__(self, img_shape, scale,
        label_noise=1e-6, loss_weights=[1, 1e-7]):
        """
        Args:
            img_shape: input image shape
            scale: image resize factor
            label_noise: noise of discriminator labels
            loss_weights: generator and discriminator loss weights
        """

        # initialize the super-class
        super(SRGAN, self).__init__(name='srgan')
        # set attributes
        self.img_shape = img_shape
        self.scale = scale
        self.label_noise = label_noise
        self.loss_weights = loss_weights

        # build the generator
        self.generator = Generator(img_shape, scale)
        # build the discriminator
        self.discriminator = Discriminator(img_shape, scale)
        # build the GAN network
        self.__build__()

    def __build__(self):
        """
        Build the GAN network
        """

        inputs = tf.keras.Input(shape=self.img_shape)
        g_outputs = self.generator(inputs)
        d_outputs = self.discriminator(g_outputs)
        super().__init__(inputs=inputs, outputs=[g_outputs, d_outputs])

    def summary(self):
        """
        Print the layer summary
        """

        print('GENERATOR MODEL')
        self.generator.summary()
        print('DISCRIMINATOR MODEL')
        self.discriminator.summary()
        print('GAN MODEL')
        super().summary()

    def compile(self, loss, optimizer, metrics=None):
        """
        Compile the model

        Args:
            loss: loss function
            optimizer: optimizing function
            metrics: evaluated metrics
        """

        # set a generator loss
        g_loss = loss
        # set a discriminator loss
        d_loss = 'binary_crossentropy'

        # compile the discriminator model
        self.discriminator.compile(loss=d_loss, optimizer=optimizer)
        # freeze discriminator weights. there are two means of freezing all the weights:
        #   (1) model.trainable = False before compiling the model
        #   (2) for layer in model.layers: layer.trainable = False - works before & after compiling
        self.discriminator.trainable = False
        # compile the GAN model
        super().compile(
            loss=[g_loss, d_loss],
            loss_weights=self.loss_weights,
            optimizer=optimizer,
            metrics={ 'g':metrics })

    def sample_labels(self, batch_size, noise):
        """
        Sample real and fake labels

        Args:
            batch_size: batch size
            noise: label noise

        Returns:
            real adn fake labels
        """
        real_labels =  np.ones(batch_size) - noise * np.random.random_sample(batch_size)
        fake_labels = np.zeros(batch_size) + noise * np.random.random_sample(batch_size)
        return real_labels, fake_labels

    def train_on_batch(self, lr_imgs, hr_imgs, sample_weight=None,
        class_weight=None, reset_metrics=True):
        """
        Runs a single gradient update on a single batch of data

        Args:
            lr_imgs: numpy array of training data (low-resolution images)
            hr_imgs: numpy array of target data (high-resolution images)
            sample_weight: optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
            class_weight: optional dictionary mapping
                class indices (integers) to a weight (float)
                to apply to the model's loss for the samples from this class during training.
            reset_metrics: If `True`, the metrics returned will be only for this
                batch. If `False`, the metrics will be statefully accumulated across batches.

        Returns:
            list of loss and metrics
        """

        # get the batch size
        batch_size = len(lr_imgs)

        # get super-resolution images using the generator
        sr_imgs = self.generator.predict_on_batch(lr_imgs)
        # get the real and fake labels with noises
        real_labels, fake_labels = self.sample_labels(
            batch_size=batch_size, noise=self.label_noise)
        # train the discriminator
        self.discriminator.train_on_batch(hr_imgs, real_labels)
        self.discriminator.train_on_batch(sr_imgs, fake_labels)

        # train the generator using the GAN model
        return super().train_on_batch(lr_imgs, [hr_imgs, real_labels])

    def test_on_batch(self, lr_imgs, hr_imgs, sample_weight=None, reset_metrics=True):
        """
        Test the model on a single batch of samples

        Args:
            lr_imgs: numpy array of training data (low-resolution images)
            hr_imgs: numpy array of target data (high-resolution images)
            sample_weight: optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
            reset_metrics: If `True`, the metrics returned will be only for this
                batch. If `False`, the metrics will be statefully accumulated across batches.

        Returns:
            list of loss and metrics
        """

        # get the batch size
        batch_size = len(lr_imgs)

        # get the real labels without a noise
        real_labels, _ = self.sample_labels(
            batch_size=batch_size, noise=self.label_noise)
        # test the GAN model
        return super().test_on_batch(lr_imgs, [hr_imgs, real_labels])

    def predict(self, x, batch_size=None, verbose=0, steps=None):
        """
        Generates output predictions for the input samples (self.generator is used as the predictor)

        Args:
            x: input low-resolution images
            batch_size: number of samples per gradient update
            verbose: verbosity mode, 0 (silent) or 1 (printing)
            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.

        Returns:
            Numpy array(s) of predictions.
        """

        return self.generator.predict(
            x, batch_size=batch_size, verbose=verbose, steps=steps)
