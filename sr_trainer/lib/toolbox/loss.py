import tensorflow as tf
from .metric import psnr,dssim

class Loss:
    """
    Define custom loss functions
    """

    @classmethod
    def psnr(cls, y_true, y_pred):
        """
        Compute inverse of psnr

        Args:
            y_true: true image
            y_pred: predicted image

        Returns:
            inverse of psnr
        """
        return 1.0 / psnr(y_true, y_pred)

    @classmethod
    def dssim(cls, y_true, y_pred):
        """
        Compute dssim = (1 - ssim) / 2

        Args:
            y_true: true image
            y_pred: predicted image

        Returns:
            dssim = (1 - ssim) / 2
        """
        return dssim(y_true, y_pred)

    @classmethod
    def vgg(cls, img_size, feature_index=20):
        """
        Compute VGG19 feature loss

        Args:
            img_size: image size with shape (width, height)
            feature_index: output layer index

        Returns:
            VGG19 feature loss function
        """

        def __preprocess():
            """
            Define a preprocessing layer for VGG19

            Returns:
                preprocessing layer for VGG19
            """
            return tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg19.preprocess_input(
                255. * tf.image.grayscale_to_rgb(x)))

        def penalty_l1(x, alpha=0.001):
            """
            Compute L1 penalty

            Args:
                alpha: penalty weight

            Returns:
                L1 penalty
            """
            return alpha * tf.keras.backend.sum(
                tf.keras.backend.maximum(0.0, tf.keras.backend.abs(x - 0.5) - 0.5))

        def penalty_l2(x, alpha=0.001):
            """
            Compute L2 penalty

            Args:
                alpha: penalty weight

            Returns:
                L2 penalty
            """
            return alpha * tf.keras.backend.sum(
                tf.keras.backend.square(tf.keras.backend.maximum(0.0, tf.keras.backend.abs(x - 0.5) - 0.5)))

        def f(y_true, y_pred, alpha=1e-4):
            """
            Comput VGG19 feature loss

            Args:
                y_true: true image
                y_pred: predicted image
                alpha: penalty coefficient

            Returns:
                VGG19 feature loss
            """
            # load VGG19 model
            vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet',
                input_shape=(*img_size, 3))
            # get a feature model
            model = tf.keras.Model(
                inputs=vgg.input,
                outputs=vgg.get_layer(index=feature_index).output,
                name='VGG19')
            # compute vgg loss
            vgg_true = model(__preprocess()(y_true))
            vgg_pred = model(__preprocess()(y_pred))
            vgg_loss = tf.keras.backend.mean(tf.keras.backend.square(vgg_true - vgg_pred))
            # add a penalty to keep within a proper range (0-1)
            return vgg_loss + penalty_l1(y_pred)

        return f
