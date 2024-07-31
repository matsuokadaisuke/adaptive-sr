import tensorflow as tf

def psnr(y_true, y_pred):
    """
    Compute peak signal-to-noise ratio (psnr)

    Args:
        y_true: true image
        y_pred: predicted image

    Returns:
        peak signal-to-noise ratio (psnr)
    """
    return tf.math.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

def dssim(y_true, y_pred):
    """
    Compute structural dissimilarity (dssim)

    Args:
        y_true: true image
        y_pred: predicted image

    Returns:
        structural dissimilarity (dssim)
    """
    ssim = tf.math.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return (1.0 - ssim) / 2.0
