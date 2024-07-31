import tensorflow as tf
import numpy as np
from .user_params import UserParams
from .toolbox.normalizer import MinMaxNormalizer
from .toolbox.data_loader import DataLoader
from .toolbox.data_generator import DataGenerator
from .toolbox.evaluator import Evaluator
from pathlib import Path

class Tester:
    """
    Test super-resolution model 

    Attributes:
        u_params: user parameters
        normalizer: normalizer
        test_data_generator: test data generator
        evaluator: trained model evaluator
    """

    def __init__(self, u_params: UserParams):
        """
        Args:
            u_params: user parameters
        """

        # set user parameters
        self.u_params = u_params
        # log directoyr
        self.log_dir = self.u_params.log_dir
        if self.log_dir is None:
            # self.log_dir = str(Path(self.u_params.fname).parent)
            p = Path(self.u_params.fname)
            self.log_dir = p.parent / (p.stem + '_log')
        # set normalizer
        self.normalizer = MinMaxNormalizer(
            xmin=self.u_params.norm_range[0], xmax=self.u_params.norm_range[1], ymin=self.u_params.norm_range[0], ymax=self.u_params.norm_range[1])

        # load training and validation data
        self.loader = DataLoader(data_dir=self.u_params.data_dir, 
                    sub_dirs=['test'], sr_scale=self.u_params.scale)
        # create test data generator
        self.test_data_generator = DataGenerator(*self.loader.test(),
            batch_size=self.u_params.batch_size,
            norm_func=self.normalizer.normalize, 
            )
        # initialize evaluator
        _, y = self.test_data_generator.sample(self.u_params.batch_size)
        self.evaluator = Evaluator(img_shape=y.shape[1:])

    def test(self, model, num_samples=4):
        """
        Test keras model

        Args:
            model: keras model
            num_samples: number of test samples
        """

        # set test mode to enable dropout and batchnormalization
        tf.keras.backend.set_learning_phase(0)

        # get test samples
        x_test, y_test = self.test_data_generator.sample(num_samples)

        # plot test results
        self.evaluator.evaluate(
            model, x_test, y_test,
            batch_size=num_samples, denorm_func=self.normalizer.denormalize,
            save_fname=Path(self.log_dir) / 'test_result.png')

    def test_with_slope_range(self, model, slmin=0.0, slmax=1.5):
        tf.keras.backend.set_learning_phase(0)

        loader = DataLoader(data_dir=self.u_params.data_dir, 
                    sub_dirs=['test'], sr_scale=self.u_params.scale, normalize=True)

        out_dir = Path(self.log_dir) / 'test_reuslt'
        out_dir.mkdir(exist_ok=True)

        x_test, y_test, test_inds, slopes, x_norm_range, y_norm_range = loader.sample_with_slope_range('test', slmin, slmax)
        # print(x_test.shape)
        
        self.normalizer = MinMaxNormalizer(
            xmin=x_norm_range[0], xmax=x_norm_range[1], ymin=y_norm_range[0], ymax=y_norm_range[1])
        
        self.logs = self.evaluator.evaluate_all(
            model, x_test, y_test, test_inds, slopes,
            denorm_func_x=self.normalizer.denormalize_x,
            denorm_func_y=self.normalizer.denormalize_y,
            save_fname = out_dir / 'test_results.csv',
        )
        