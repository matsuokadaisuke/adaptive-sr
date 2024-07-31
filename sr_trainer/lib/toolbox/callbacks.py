import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import tensorflow as tf
from .plotter import Plotter
from .evaluator import Evaluator

class ModelLogger(tf.keras.callbacks.ModelCheckpoint):
    """
    Log trained models

    Attributes:
        log_dir: log output directory
        log_period: logging period
    """

    def __init__(self, log_dir, log_period):
        """
        Args:
            log_dir: log output directory
            log_period: logging period
        """

        # call super-class constructor
        super().__init__(
            filepath=str(log_dir / 'weights_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True)
        # store attributes
        self.log_dir = log_dir
        self.log_period = log_period

    def on_epoch_end(self, epoch, logs=None):
        """
        Save trained model on epoch end

        Args:
            epoch: current epoch
            logs: log data { 'loss':value, 'val_loss':value, 'psnr':value,
                             'val_psnr':value, 'dssim':value, 'val_dssim':value }
        """

        # call super-class method
        super().on_epoch_end(epoch, logs)

        # culling epochs
        if (epoch + 1) % self.log_period:
            return
        # save model
        fname = self.log_dir / 'weights_{:05d}.h5'.format(epoch + 1)
        self.model.save_weights(str(fname))

class HistoryLogger(tf.keras.callbacks.CSVLogger):
    """
    Log training history

    Attributes:
        log_dir: log output directory
        log_period: logging period
        epochs: epoch history
        logs: log data { 'loss':[], 'val_loss':[], 'psnr':[],
                         'val_psnr':[], 'dssim':[], 'val_dssim':[] }
    """

    def __init__(self, log_dir, log_period):
        """
        Args:
            log_dir: log output directory
            log_period: logging period
        """

        # call super-class constructor
        super().__init__(log_dir / 'history.csv')

        # store attributes
        self.log_dir = log_dir
        self.log_period = log_period
        # initialize attributes
        self.epochs = []
        self.logs = defaultdict(list)

    def on_epoch_end(self, epoch, logs=None):
        """
        Save history graph on epoch end

        Args:
            epoch: current epoch
            logs: log data { 'loss':value, 'val_loss':value, 'psnr':value,
                             'val_psnr':value, 'dssim':value, 'val_dssim':value }
        """

        # call super-class method
        super().on_epoch_end(epoch, logs)

        # stack epoch history
        self.epochs.append(epoch + 1)
        # stack log history
        for k, v in logs.items():
            self.logs[k].append(v)

        # culling epochs
        if (epoch + 1) % self.log_period:
            return
        # save a history graph
        fname = self.log_dir / 'history_{:05d}.png'.format(epoch + 1)
        Plotter.plot_history_graphs(epochs=self.epochs, logs=self.logs, save_fname=fname)

class TestLogger(tf.keras.callbacks.Callback):
    """
    Log test images

    Args:
        log_dir: log output directory
        log_period: logging period
        data_generator: data generator (validation)
        denorm_func: denormalization function
        num_samples: number of test samples
        evaluator: trained model evaluator
    """

    def __init__(self, log_dir, log_period,
        data_generator, denorm_func_x, denorm_func_y, num_samples):
        """
        Args:
            log_dir: log output directory
            log_period: logging period
            data_generator: data generator (validation)
            denorm_func: denormalization function
            num_samples: number of test samples
        """

        # store attributes
        self.log_dir = log_dir
        self.log_period = log_period
        self.data_generator = data_generator
        self.denorm_func_x = denorm_func_x
        self.denorm_func_y = denorm_func_y
        self.num_samples = num_samples
        # initialize evaluator
        _, y = self.data_generator.sample(self.num_samples)
        self.evaluator = Evaluator(img_shape=y.shape[1:])

    def on_epoch_end(self, epoch, logs=None):
        """
        Save test images on epoch end

        Args:
            epoch: current epoch
            logs: log data { 'loss':value, 'val_loss':value, 'psnr':value,
                             'val_psnr':value, 'dssim':value, 'val_dssim':value }
        """

        # culling epochs
        if (epoch + 1) % self.log_period:
            return

        # get test samples
        x_test, y_test = self.data_generator.sample(self.num_samples)
        # save a test image
        fname = self.log_dir / 'test_{:05d}.png'.format(epoch + 1)
        self.evaluator.evaluate(self.model, x_test, y_test,
            batch_size=self.num_samples, denorm_func_x=self.denorm_func_x, denorm_func_y=self.denorm_func_y,
            save_fname=fname)
