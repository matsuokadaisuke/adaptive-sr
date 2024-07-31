import tensorflow as tf 
from pathlib import Path

from .user_params import UserParams
from .toolbox.model.srcnn import SRCNN
from .toolbox.model.fsrcnn import FSRCNN
from .toolbox.model.espcn import ESPCN
from .toolbox.model.srgan import SRGAN
from .toolbox.model.esrgan import ESRGAN
from .toolbox.loss import Loss
from .toolbox.metric import psnr,dssim

class Modeler:
    """
    Build super resolution model using keras

    Attributes:
        u_params: user parameters
        models: selectable models
        loss_funcs: loss functions
        optimizers: optimizers
    """

    def __init__(self, u_params: UserParams):
        """
        Args:
            u_params: user parameters
        """

        # set user parameters
        self.u_params = u_params
        # set selectable models
        self.models = {
            'srcnn': SRCNN,
            'fsrcnn': FSRCNN,
            'espcn': ESPCN,
            'srgan': SRGAN,
            'esrgan': ESRGAN,
        }
        # set loss functions
        self.loss_funcs = {
              'mae': tf.keras.losses.MAE,
              'mse': tf.keras.losses.MSE,
             'psnr': Loss.psnr,
            'dssim': Loss.dssim,
             'vgg' : Loss.vgg(img_size=self.u_params.input_shape[:-1]),
        }
        # set optimizers
        self.optimizers = {
                 'sgd': tf.keras.optimizers.SGD,
             'rmsprop': tf.keras.optimizers.RMSprop,
             'adagrad': tf.keras.optimizers.Adagrad,
            'adadelta': tf.keras.optimizers.Adadelta,
                'adam': tf.keras.optimizers.Adam,
              'adamax': tf.keras.optimizers.Adamax,
               'nadam': tf.keras.optimizers.Nadam,
        }

    def build(self, verbose=1, mode='train'):
        """
        build a model

        Args:
            verbose: If 1, model summary is printed (default is 1)

        Returns:
            built model
        """
        # create new model
        print('building model...', end='', flush=True)
        model = self.models[self.u_params.model](
           img_shape=self.u_params.input_shape, scale=self.u_params.scale, params=self.u_params)
            
        print(' done')
        # load weights
        if Path(self.u_params.fname_model_weights).exists():
            print('loading weights...', end='', flush=True)
            print(self.u_params.fname_model_weights)
            model.load_weights(self.u_params.fname_model_weights)
            print(' done')
        else:
            if mode == 'test':
                # fname_weights = Path(self.u_params.fname).parent / 'weights_best.h5'
                p = Path(self.u_params.fname)
                log_dir = p.parent / (p.stem + '_log')
                fname_weights = log_dir /  'weights_best.h5'
                if fname_weights.exists():
                    print('loading weights for test...', end='', flush=True)
                    print(str(fname_weights))
                    model.load_weights(fname_weights)
                    print(' done')
                else:
                    print('error : could not find weights file in test mode.')
                    return None

        # compile model
        model.compile(
            loss=self.loss_funcs[self.u_params.loss],
            optimizer=self.optimizers[self.u_params.optimizer](
                lr=self.u_params.learning_rate),
            metrics=[psnr, dssim]
        )
        # print summary
        if verbose > 0:
            model.summary()

        return model
