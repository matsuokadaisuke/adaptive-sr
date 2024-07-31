import os, warnings
import numpy as np
import tensorflow as tf

class Initializer:
    """
    Initialize systems
    """

    @classmethod
    def tf_init(cls, log_level='3', seed=None):
        """
        Initialize numpy and tensorflow status

        Args:
            log_level: log level (0: all log, 1: warnings and errors, 2: errors, 3: no log)
            seed: random seed value
        """
        # hide redundant tensorflow warnings
        warnings.filterwarnings('ignore')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = log_level
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	    # initialize cudnn
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        # initialize random seeds
        if seed != None:
            np.random.seed(seed)
            tf.compat.v1.set_random_seed(seed)
		# set the session
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(
            graph=tf.compat.v1.get_default_graph(), config=tf.compat.v1.ConfigProto()))