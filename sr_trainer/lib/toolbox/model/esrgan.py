import numpy as np
import tensorflow as tf
from .layer import Resize, Conv2DSubPixel, Scale

class Generator(tf.keras.Model):
    """
    Build the generator network of ESRGAN (see https://arxiv.org/abs/1809.00219)

    Attributes:
        img_shape: input image shape (width, height, channels)
        scale: image resize factor
        filters: 畳み込み層のチャネル数
        num_residual_blocks: RRDBブロック数
    """

    def __init__(self, img_shape, scale, filters, num_residual_blocks):
        """
        Args:
            img_shape: input low-resolution image shape
            scale: image resize factor
            filters: 畳み込み層のチャネル数
            num_residual_blocks: RRDBブロック数
        """

        # initialize the super-class
        super(Generator, self).__init__(name='generator')
        # set attributes
        self.img_shape = img_shape
        self.scale = scale
        # build a network
        self.filters = filters
        self.num_residual_blocks = num_residual_blocks
        self.__build__(filters=self.filters, num_residual_blocks=self.num_residual_blocks)

    def __DB(self, filters, kernel_initializer, alpha, kernel_size=3, strides=1):
        """
        Dense block (DB)

        Args:
            filters: number of kernels
            kernel_initializer: kernel initializer
            alpha: LeakyReLU slope coefficient
            kernel_size: height and width of the 2D convolution window
            strides: strides of the convolution along the height and width

        Returns:
            dense block layer
        """
        def f(input):
            layers = [ input ]
            for _ in range(4):
                x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer=kernel_initializer,
                    strides=strides, padding='same')(layers[-1])
                x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)
                x = tf.keras.layers.Concatenate()([l for l in layers] + [x])
                layers.append(x)
            return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                kernel_initializer=kernel_initializer,
                strides=strides, padding='same')(layers[-1])
        return f

    def __RRDB(self, filters, kernel_initializer, alpha, beta):
        """
        Residual-in-Residual Dense block (RRDB)

        Args:
            filters: number of kernels
            kernel_initializer: kernel initializer
            alpha: LeakyReLU slope coefficient
            beta: residual scaling factor

        Returns:
            residual block layer
        """
        def f(input):
            x = input
            for _ in range(3):
                y = self.__DB(filters, kernel_initializer, alpha)(x)
                y = Scale(beta)(y)
                x = tf.keras.layers.Add()([x, y])
            x = Scale(beta)(x)
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
            return tf.keras.layers.PReLU(shared_axes=[1, 2],
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

    def __build__(self, filters=32, num_residual_blocks=2,
        kernel_initializer='he_normal', alpha=0.2, beta=0.2):
        """
        Build the generator network of ESRGAN

        Args:
            filters: base number of kernels
            num_residual_blocks: number of residual blocks
            kernel_initializer: kernel initializer
            alpha: LeakyReLU slope coefficient
            beta: residual scaling factor
        """
        # input low-resolution image
        inputs = tf.keras.Input(shape=self.img_shape)

        # pre-residual block
        x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
            kernel_initializer=kernel_initializer)(inputs)
        x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)
        # residual-inresidual dense block
        r = x
        for _ in range(num_residual_blocks):
            r = self.__RRDB(filters, kernel_initializer, alpha, beta)(x)
        # post-residual block
        r = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
            kernel_initializer=kernel_initializer)(r)
        x = tf.keras.layers.Add()([x, r])
        # upsampling blocks
        num_upsampling_blocks = int(np.log2(self.scale))
        for _ in range(num_upsampling_blocks):
            x = self.__upsampling_block(
                filters, kernel_initializer=kernel_initializer)(x)

        # final convolution
        x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
            kernel_initializer=kernel_initializer)(x)
        x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)

        # output high-resolution image
        outputs = tf.keras.layers.Conv2D(self.img_shape[-1],
            kernel_size=3, strides=1, padding='same',
            kernel_initializer=kernel_initializer)(x)

        # setup the network
        super().__init__(inputs=inputs, outputs=outputs, name='generator')

class Discriminator(tf.keras.Model):
    """
    Build the discriminator network

    Attributes:
        img_shape: input image shape (width, height, channels)
        scale: image resize factor
        d_filters: number of kernels
        d_num_downsampling_blocks: number of downsampling blocks
    """

    def __init__(self, img_shape, scale, d_filters=4, d_num_downsampling_blocks=2):
        """
        Args:
            img_shape: input low-resolution image shape
            scale: image resize factor
            d_filters: number of kernels
            d_num_downsampling_blocks: number of downsampling blocks
        """

        # initialize the super-class
        super(Discriminator, self).__init__(name='discriminator')
        # set attributes
        self.img_shape = img_shape
        self.scale = scale
        # build a network
        self.filters = d_filters
        self.num_downsampling_blocks = d_num_downsampling_blocks
        self.__build__(filters=self.filters, num_downsampling_blocks=self.num_downsampling_blocks)

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
                x = tf.keras.layers.BatchNormalization(momentum=momentum, fused=False)(x)
            return tf.keras.layers.LeakyReLU(alpha=alpha)(x)
        return f

    def __build__(self, filters=4, num_downsampling_blocks=2,
        kernel_initializer='he_normal', momentum=0.8, alpha=0.2, dropout_rate=0.4):
        """
        Build the discriminator network of ESRGAN

        Args:
            filters: base number of kernels
            num_downsampling_blocks: number of downsampling blocks
            kernel_initializer: kernel initializer
            momentum: momentum for the moving mean and the moving variance
            alpha: LeakyReLU slope coefficient
            dropout_rate: 
        """

        # get hig-resolution image shape
        hr_img_shape = (self.scale * self.img_shape[0],
                        self.scale * self.img_shape[1], self.img_shape[2])
        # input layer
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
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

        # output the real or fake label
        outputs = tf.keras.layers.Dense(1, kernel_initializer=kernel_initializer)(x)

        # setup the network
        super().__init__(inputs=inputs, outputs=outputs, name='discriminator')

class RelativisticDiscrimiantor(tf.keras.Model):
    """
    Build the relativistic discriminator network of ESRGAN (see https://arxiv.org/abs/1809.00219)

    Attributes:
        img_shape: input image shape (width, height, channels)
        scale: image resize factor
        d_filters: number of kernels
        d_num_downsampling_blocks: number of downsampling blocks
    """

    def __init__(self, img_shape, scale, d_filters=4, d_num_downsampling_blocks=2):
        """
        Args:
            img_shape: input low-resolution image shape
            scale: image resize factor
            d_filters: number of kernels
            d_num_downsampling_blocks: number of downsampling blocks
        """

        # initialize the super-class
        super(RelativisticDiscrimiantor, self).__init__(name='relativistic_discriminator')
        # set attributes
        self.img_shape = img_shape
        self.scale = scale
        # build a network
        self.d_filters = d_filters
        self.d_num_downsampling_blocks = d_num_downsampling_blocks
        self.__build__()

    def __ra_loss(self, x, noise=1e-4):
        """
        Compute relativistic average loss

        Args:
            x: real and fake labels
            noise: stabilizing noise to avoid to diverge log-terms

        Returns:
            relativistic average loss
        """
        real_label, fake_label = x
        d_real = tf.math.log_sigmoid(real_label - tf.reduce_mean(fake_label))
        # 1-sigmoid(x) = sigmoid(-x)
        d_fake = tf.math.log_sigmoid(-(fake_label - tf.reduce_mean(real_label)))
        return - d_real - d_fake

    def __build__(self):
        """
        Build the discriminator network of ESRGAN
        """

        # get hig-resolution image shape
        hr_img_shape = (self.scale * self.img_shape[0],
                        self.scale * self.img_shape[1], self.img_shape[2])
        # input layers
        hr_img = tf.keras.Input(shape=hr_img_shape, name='input_real')
        sr_img = tf.keras.Input(shape=hr_img_shape, name='input_fake')
        # build the discriminator network
        discriminator = Discriminator(self.img_shape, self.scale, self.d_filters, self.d_num_downsampling_blocks)
        real_label = discriminator(hr_img)
        fake_label = discriminator(sr_img)
        # compute the relativistic average loss
        loss = tf.keras.layers.Lambda(self.__ra_loss, name='ra_loss')([real_label, fake_label])

        # setup the network
        super().__init__(inputs=[hr_img, sr_img], outputs=loss, name='discriminator')

class ESRGAN(tf.keras.Model):
    """
    Build the ESRGAN network (see https://arxiv.org/abs/1809.00219)

    Attributes:
        img_shape: input image shape (width, height, channels)
        scale: image resize factor
        loss_weights: generator and discriminator loss weights
        params: read yaml(default is None)
        generator: generator network
        discriminator: relativistic discriminator network
    """

    def __init__(self, img_shape, scale, loss_weights=[1, 1.0e-7], params=None):
    # def __init__(self, img_shape, scale, loss_weights=[1, 1e-7], filters=32, num_residual_blocks=2):
        """
        Args:
            img_shape: input image shape
            scale: image resize factor
            loss_weights: generator and discriminator loss weights
            params: read yaml(default is None)
        """

        # initialize the super-class
        super(ESRGAN, self).__init__(name='esrgan')
        # set attributes
        self.img_shape = img_shape
        self.scale = scale
        self.params = params
        self.loss_weights = loss_weights
        if self.params is not None:
            if hasattr(self.params, 'esrgan'):
                if 'loss_weights' in self.params.esrgan:
                    # work around for 1e-7 in yml. 1.0e-7 is what we need.
                    lw = [float(v) for v in self.params.esrgan['loss_weights']]
                    self.loss_weights = lw
        print('build esrgan lw={}'.format(self.loss_weights))

        self.filters = 32
        self.num_residual_blocks = 2
        if self.params is not None:
            if hasattr(self.params, 'esrgan'):
                self.filters = self.params.esrgan['filters']
                self.num_residual_blocks = self.params.esrgan['num_residual_blocks']
        # build the generator
        print('build esrgan gf={}, gr={}'.format(self.filters, self.num_residual_blocks))
        self.generator = Generator(img_shape, scale, self.filters, self.num_residual_blocks)
        # build the relativistic discriminator
        self.d_filters = 4
        self.d_num_downsampling_blocks = 2
        if self.params is not None:
            if hasattr(self.params, 'esrgan'):
                if 'd_filters' in self.params.esrgan:
                    self.d_filters = self.params.esrgan['d_filters']
                    self.d_num_downsampling_blocks = self.params.esrgan['d_num_downsampling_blocks']
        print('build esrgan df={}, dn={}'.format(self.d_filters, self.d_num_downsampling_blocks))
        self.discriminator = RelativisticDiscrimiantor(img_shape, scale, self.d_filters, self.d_num_downsampling_blocks)
        # build the GAN network
        self.__build__()

    def __build__(self):
        """
        Build the GAN network
        """

        # set low-resolution input layer
        lr_img = tf.keras.Input(shape=self.img_shape, name='input_lr')
        # get a super-resolution image
        sr_img = self.generator(lr_img)

        # get hig-resolution image shape
        hr_img_shape = (self.scale * self.img_shape[0],
                        self.scale * self.img_shape[1], self.img_shape[2])
        # set high-resolution input layer
        hr_img = tf.keras.Input(shape=hr_img_shape, name='input_hr')
        # compute relativistic average loss via the discriminator (hr_img and sr_img are reversed)
        loss = tf.keras.layers.Lambda(lambda x: self.discriminator(x),
            name='relativistic_discriminator')([sr_img, hr_img])

        # build the network
        super().__init__(inputs=[lr_img, hr_img], outputs=loss)

    def summary(self):
        """
        Print the layer summary
        """

        print('GENERATOR MODEL')
        self.generator.summary()
        print('RELATIVISTIC DISCRIMINATOR MODEL')
        self.discriminator.summary()
        print('GAN MODEL')
        super().summary()

    def compile(self, loss, optimizer, metrics=None):
        """
        Compile the model

        Args:
            loss: image loss function
            optimizer: optimizing function
            metrics: evaluated metrics
        """

        # set the discriminator loss
        self.discriminator.add_loss(
            tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x))(
            #self.discriminator.get_layer(index=-1).output))
            self.discriminator.get_layer('ra_loss').output))
        # compile the discriminator model
        self.discriminator.compile(
            optimizer=optimizer)

        # get the high-resolution image layer
        hr_img = self.get_layer('input_hr').output
        # get the super-resolution image layer
        sr_img = self.get_layer('generator').get_output_at(-1)
        # compute a generator (perceptual) loss
        g_loss = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(loss(x[0], x[1])))([hr_img, sr_img])
        # compute a discriminator (adversarial) loss
        d_loss = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x))(self.get_layer('relativistic_discriminator').get_output_at(-1))
        # compute a total loss
        t_loss = tf.keras.layers.Lambda(lambda x:
            self.loss_weights[0]*x[0] + self.loss_weights[1]*x[1])([g_loss, d_loss])

        # add the total loss
        self.add_loss(t_loss)
        # freeze discriminator weights. there are two means of freezing all the weights:
        #   (1) model.trainable = False before compiling the model
        #   (2) for layer in model.layers: layer.trainable = False - works before & after compiling
        self.discriminator.trainable = False
        # compile the GAN model
        super().compile(optimizer=optimizer)

        # add logging losses
        self.add_metric(g_loss, name='g_loss', aggregation='mean')
        self.add_metric(d_loss, name='d_loss', aggregation='mean')
        # add logging metrics
        for metric in metrics:
            tensors = metric(hr_img, sr_img)
            self.add_metric(tensors, name='g_' + metric.__name__, aggregation='mean')

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

        # get super-resolution images using the generator
        sr_imgs = self.generator.predict_on_batch(lr_imgs)
        # train the discriminator
        loss1 = self.discriminator.train_on_batch([hr_imgs, sr_imgs], None)
        # train the generator using the GAN model
        loss2 = super().train_on_batch([lr_imgs, hr_imgs], None)
        return loss2

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

        return super().test_on_batch([lr_imgs, hr_imgs], None)

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
