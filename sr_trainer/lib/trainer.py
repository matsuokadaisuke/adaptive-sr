import tensorflow as tf
from pathlib import Path
from .user_params import UserParams
from .toolbox.normalizer import MinMaxNormalizer
from .toolbox.data_loader import DataLoader
from .toolbox.data_generator import DataGenerator
from .toolbox.callbacks import ModelLogger, HistoryLogger, TestLogger
import tensorflow as tf
import shutil

class Trainer:
    """
    Train super-resolution model

    Attributes:
        u_params: user parameters
        normalizer: normalizer
        train_data_generator: training data generator
        validation_data_generator: validation data generator
    """
    def __init__(self, u_params: UserParams):
        """
        Args:
            u_params: user parameters
        """

        # set user parameters
        self.u_params = u_params
        # set normalizer
        
        self.normalizer = MinMaxNormalizer(
            xmin=self.u_params.norm_range[0], xmax=self.u_params.norm_range[1], ymin=self.u_params.norm_range[0], ymax=self.u_params.norm_range[1])

        # load training and validation data
        loader = DataLoader(data_dir=self.u_params.data_dir,
            sub_dirs=['train', 'validation'], sr_scale=self.u_params.scale, normalize=True)
        
        x_norm_range, y_norm_range = loader.return_norm_range()
        
        self.normalizer = MinMaxNormalizer(
            xmin=x_norm_range[0], xmax=x_norm_range[1], ymin=y_norm_range[0], ymax=y_norm_range[1])
        
        # create training data generator
        if hasattr(self.u_params, 'online_da'):
            online_da = self.u_params.online_da
        else:
            online_da = None
        self.train_data_generator = DataGenerator(*loader.train(),
            batch_size = self.u_params.batch_size,
            norm_func=self.normalizer.normalize,
            da_conf=online_da,
            )            
        # create validation data generator
        self.validation_data_generator = DataGenerator(*loader.validation(),
            batch_size = self.u_params.batch_size,
            norm_func=self.normalizer.normalize)       

    def reset_weights(self, model):
        """
        reset all weights

        Args:
            model: keras model
        """
        session = tf.compat.v1.keras.backend.get_session()
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
            if hasattr(layer, 'bias_initializer'):
                layer.bias.initializer.run(session=session)

    def train(self, model):
        """
        Train keras model

        Args:
            model: keras model
        """
        # set training mode to enable dropout and batchnormalization
        tf.keras.backend.set_learning_phase(1)

        # make log directory    
        if self.u_params.log_dir is None:
            p = Path(self.u_params.fname)
            log_dir = p.parent / (p.stem + '_log')
        else:
            log_dir = Path(self.u_params.log_dir)
        if not log_dir.exists():
            log_dir.mkdir()
        shutil.copy(self.u_params.fname, log_dir)
        print(str(log_dir))

        # eary stopping
        if self.u_params.patience is None:
            # set callback functions
            callbacks = [
                ModelLogger(log_dir=log_dir, log_period=self.u_params.log_period),
                HistoryLogger(log_dir=log_dir, log_period=self.u_params.log_period),
                TestLogger(log_dir=log_dir, log_period=self.u_params.log_period,
                    data_generator=self.validation_data_generator,
                    denorm_func_x=self.normalizer.denormalize_x,
                    denorm_func_y=self.normalizer.denormalize_y,
                    num_samples=4),
            ]
        else:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.u_params.patience, restore_best_weights=True)
            # set callback functions
            callbacks = [
                ModelLogger(log_dir=log_dir, log_period=self.u_params.log_period),
                HistoryLogger(log_dir=log_dir, log_period=self.u_params.log_period),
                TestLogger(log_dir=log_dir, log_period=self.u_params.log_period,
                    data_generator=self.validation_data_generator,
                    denorm_func_x=self.normalizer.denormalize_x,
                    denorm_func_y=self.normalizer.denormalize_y,
                    num_samples=4),
                early_stopping,
            ]
        train_gen = self.train_data_generator.generator()
        steps_per_epoch_mag = 1
        if hasattr(self.u_params, 'online_da'):
            if self.u_params.online_da['valid']:
                train_gen = self.train_data_generator.generator_with_da()
                steps_per_epoch_mag = self.u_params.online_da['steps_per_epoch_mag']

        model.fit_generator(
            # generator=self.train_data_generator.generator(),
            # generator=self.train_data_generator.generator_with_da(),
            generator = train_gen,
            steps_per_epoch=self.train_data_generator.steps_per_epoch() * steps_per_epoch_mag,
            epochs=self.u_params.epochs,
            validation_data=self.validation_data_generator.generator(),
            validation_steps=self.validation_data_generator.steps_per_epoch(),
            callbacks=callbacks)

        score = model.evaluate_generator(self.validation_data_generator.generator(),
                                        steps=self.validation_data_generator.steps_per_epoch())
        return score