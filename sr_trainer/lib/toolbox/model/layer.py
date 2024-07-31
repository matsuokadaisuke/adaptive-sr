import tensorflow as tf

class Resize(tf.keras.layers.Layer):
    """
    Resizing image layer

    Attributes:
        scale: resize scale
        method: interpolation method (BILINEAR, BICUBIC, ...)
    """

    def __init__(self, scale, method=tf.image.ResizeMethod.BICUBIC,
        trainable=False, margin=0, **kwargs):
        """
        Args:
            scale: resize scale
            method: interpolation method (BILINEAR, BICUBIC, ...)
            trainable: trainability (default is False)
            kwargs: other arguments
        """
        self.scale = scale
        self.margin = margin
        self.method = method
        super().__init__(trainable=trainable, **kwargs)

    def resized_shape(self, input_shape):
        """
        Get resized shape

        Args:
            input_shape: input shape

        Returns:
            resized shape
        """
        return tuple([ self.scale * l + self.margin for l in input_shape[1:-1] ])

    def call(self, x):
        """
        Resize image

        Args:
            x: input image

        Returns:
            resizing logic
        """
        return tf.image.resize(
            x, self.resized_shape(x.shape), method=self.method)

    def compute_output_shape(self, input_shape):
        """
        Compute output shape

        Args:
            input_shape: input shape

        Returns:
            layer output shape
        """
        return (input_shape[0], *self.resized_shape(input_shape), input_shape[-1])

    def get_config(self):
        """
        Get layer configuration as a dictionary

        Returns:
            layer configuration
        """
        config = super().get_config()
        config['scale'] = self.scale
        config['method'] = self.method
        return config

class Conv2DSubPixel(tf.keras.layers.Layer):
    """
    Sub-pixel convolution layer

    Attributes:
        scale: resize scale
    """

    def __init__(self, scale, trainable=False, **kwargs):
        """
        Args:
            scale: resize scale
            trainable: trainability (default is False)
            kwargs: other arguments
        """
        self.scale = scale
        super().__init__(trainable=trainable, **kwargs)

    def call(self, x):
        """
        sub-pixel convolution, where depth_to_space(x, b) changes
        shape (B, H, W, C) to (B, Hb, Wb, C/b^2)

        Args:
            x: input image

        Returns:
            sub-pixel convolution logic
        """
        return tf.nn.depth_to_space(x, self.scale)

    def compute_output_shape(self, input_shape):
        """
        Compute output shape

        Args:
            input_shape: input shape

        Returns:
            layer output shape
        """
        return (input_shape[0],
                input_shape[1] * self.scale,
                input_shape[2] * self.scale,
                int(input_shape[3] / (self.scale ** 2)))

    def get_config(self):
        """
        Get layer configuration as a dictionary

        Returns:
            layer configuration
        """
        config = super().get_config()
        config['scale'] = self.scale
        return config

class Scale(tf.keras.layers.Layer):
    """
    scaling layer

    Attributes:
        factor: scaling factor
    """

    def __init__(self, factor, trainable=False, **kwargs):
        """
        Args:
            factor: scaling factor
            trainable: trainability (default is False)
            kwargs: other arguments
        """
        self.factor = factor
        super().__init__(trainable=trainable, **kwargs)

    def call(self, x):
        """
        Scaling layer

        Args:
            x: input layer

        Returns:
            scaling logic
        """
        return tf.keras.layers.Lambda(lambda l: self.factor * l)(x)

    def compute_output_shape(self, input_shape):
        """
        Compute output shape

        Args:
            input_shape: input shape

        Returns:
            layer output shape
        """
        return input_shape

    def get_config(self):
        """
        Get layer configuration as a dictionary

        Returns:
            layer configuration
        """
        config = super().get_config()
        config['factor'] = self.factor
        return config
