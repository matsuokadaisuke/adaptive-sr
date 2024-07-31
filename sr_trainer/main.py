import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from lib.arg_parser import ArgParser
from lib.user_params import UserParams
from lib.initializer import Initializer
from lib.modeler import Modeler
from lib.trainer import Trainer
from lib.tester import Tester

if __name__ == '__main__':

        # parse command-line arguments
        parser = ArgParser()

        # load user parameters
        u_params = UserParams(parser.get_fname())

        # initialize numpy and tensorflow status
        if parser.get_mode() == 'train':
            Initializer.tf_init(seed=10)  # set a fixed seed in training mode
        else:
            Initializer.tf_init(seed=None)
        # build a model
        model = Modeler(u_params).build(mode=parser.get_mode())
        if model is None:
            import sys
            sys.exit()

        # set mode functions
        modes = {
            'train' : Trainer(u_params).train,
            'test'  : Tester(u_params).test_with_slope_range,
        }
        # run a mode function
        modes[parser.get_mode()](model)
