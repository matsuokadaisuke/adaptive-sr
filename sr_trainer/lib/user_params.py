import yaml

class UserParams:
    """
    Define user parameters

    Attributes:
        input_shape: base image shape. (32, 32, 1) is recommended.
        scale: super-resolution scale. 2x is recommended.
        data_dir: dataset directory including train, validation and test directories
        norm_range: data range for normalization
        fname_model_weights: filename of trained model weights (.h5)
        batch_size: training batch size
        epochs: number of training epochs
        model: model type (srcnn, fsrcnn, espcn, srgan)
        loss: loss function (mae, mse, psnr, dssim, vgg)
        optimizer: optimizer (sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam)
        learning_rate: learning rate
        log_period: logging period
        patience: ここで指定したエポック数の間（監視する値に）改善がないと，訓練が停止
    """

    def __init__(self, fname):
        """
        Register attributes from parameter dictionary

        Args:
            fname: yml filename including user parameters
        """

        # load yml file
        with open(fname, 'r') as f:
            param_dict = yaml.full_load(f)
        # set parameters to dictionary
        for k, v in param_dict.items():
            self.__dict__[k] = v
        self.fname = fname