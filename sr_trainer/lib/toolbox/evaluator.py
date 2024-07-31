import cv2
import numpy as np
import tensorflow as tf
from .metric import psnr,dssim
from .plotter import Plotter
import pandas as pd
from tensorflow.keras.metrics import mean_squared_error as mse
from tqdm import tqdm
import pickle

class Evaluator:
    """
    Evaluate a trained model for test data

    Attributes:
        X: placeholder to compute metrics
        Y: placeholder to compute metrics
        metric_ops: metric computing operators
    """

    def __init__(self, img_shape):
        """
        Args:
            img_shape: image shape (width, height, channel)
        """
        # setup placeholders and operations to accelerate metric computation
        self.X = tf.compat.v1.placeholder(tf.float32,shape=img_shape)
        self.Y = tf.compat.v1.placeholder(tf.float32,shape=img_shape)
        self.metric_ops = [ psnr(self.X, self.Y), dssim(self.X, self.Y), tf.math.sqrt(tf.math.reduce_mean(mse(self.X, self.Y))) ]

    def evaluate(self, model, x_test, y_test, batch_size,
        denorm_func_x, denorm_func_y, save_fname=None):
        """
        Evaluate a trained model for test data

        Args:
            model: trained model
            x_test: low-resolution test images
            y_test: high-resolution test images
            batch_size: number of images
            denorm_func: denormalization function
            save_fname: save filename (If None, show is called)
        """

        # compute a nearest interpolated image
        y_nrst = np.asarray([
            cv2.resize(img.squeeze(), y_test.squeeze().shape[1:],
                interpolation=cv2.INTER_NEAREST) for img in x_test])
        y_nrst = np.expand_dims(y_nrst, axis=-1)
        # compute a bicubic interpolated image
        y_intp = np.asarray([
            cv2.resize(img.squeeze(), y_test.squeeze().shape[1:],
                interpolation=cv2.INTER_CUBIC) for img in x_test])
        y_intp = np.expand_dims(y_intp, axis=-1)
        # compute a super-resolution image
        y_pred = model.predict(x_test, batch_size=batch_size)

        # setup logs with shape (num_groups, num_samples, psnr and dssim)
        logs = np.full((4, batch_size, 3), np.nan)
        # get the tensorflow session
        sess = tf.compat.v1.keras.backend.get_session()
        # evalualte psnr and dssim for low-resolution, bicubic and super-resolution images
        for i, y_cmpr in enumerate([y_nrst, y_intp, y_pred]):
            for j, (x, y) in enumerate(zip(y_cmpr, y_test)):
                logs[i, j] = [ sess.run(op, feed_dict={ self.X:x, self.Y:y })
                    for op in self.metric_ops]

        # denormalize images
        x_test = denorm_func_x(x_test)
        y_test = denorm_func_y(y_test)
        y_intp = denorm_func_x(y_intp)
        y_pred = denorm_func_y(y_pred)

        # plot a test image
        Plotter.plot_test_imgs(
            imgs=[x_test, y_intp, y_pred, y_test],
            labels=['low-res', 'bicubic', 'super-res', 'high-res'],
            logs=logs,
            save_fname=save_fname)

    def evaluate_all(self, model, x_test, y_test, test_indices, slopes,
        denorm_func_x, denorm_func_y, save_fname):

        gen_plot = True
        save_all = True

        # compute a nearest interpolated image
        y_nrst = np.asarray([
            cv2.resize(img.squeeze(), y_test.squeeze().shape[1:],
                interpolation=cv2.INTER_NEAREST) for img in x_test])
        y_nrst = np.expand_dims(y_nrst, axis=-1)
        # compute a bicubic interpolated image
        y_intp = np.asarray([
            cv2.resize(img.squeeze(), y_test.squeeze().shape[1:],
                interpolation=cv2.INTER_CUBIC) for img in x_test])
        y_intp = np.expand_dims(y_intp, axis=-1)
        # compute a super-resolution image
        y_pred = model.predict(x_test)

        # setup logs with shape (num_groups, num_samples, psnr and dssim)
        logs = np.full((4, len(x_test), 3), np.nan)
        # get the tensorflow session
        sess = tf.compat.v1.keras.backend.get_session()
        # evalualte psnr and dssim for low-resolution, bicubic and super-resolution images
        log_list = []
        case_labels = ['low-res_', 'bicubic_', 'super-res_']
        for i, y_cmpr in enumerate([y_nrst, y_intp, y_pred]):
            mts = []
            for j, (x, y, index, slope) in tqdm(enumerate(zip(y_cmpr, y_test, test_indices, slopes)), desc='calc metrics for {}'.format(case_labels[i])):
                res = [ sess.run(op, feed_dict={ self.X:x, self.Y:y })
                     for op in self.metric_ops]
                logs[i, j] = res
                mts.append({
                    # 'i': i,
                    'img_index': index,
                    'mean_slope_gradient': slope,
                    case_labels[i] + 'psnr':res[0],
                    case_labels[i] + 'dssim':res[1],
                    case_labels[i] + 'rmse':res[2]
                })
            log_list.append(pd.DataFrame(mts))

        logs_df = pd.merge(log_list[0], log_list[1], on=['img_index', 'mean_slope_gradient'])
        logs_df = pd.merge(logs_df, log_list[2], on=['img_index', 'mean_slope_gradient'])
        # print(y_pred.shape)
        # logs_df = pd.DataFrame(logs_1)
        logs_df.to_csv(save_fname, index=None)
        logs_df.describe().to_csv(save_fname.with_name(save_fname.stem + '_describe.csv'))
        logs_df[ (logs_df.mean_slope_gradient >= 0.0) & (logs_df.mean_slope_gradient < 0.05) ].describe().to_csv(save_fname.with_name(save_fname.stem + 'slope0-0.05_describe.csv'))
        logs_df[ (logs_df.mean_slope_gradient >= 0.05) & (logs_df.mean_slope_gradient < 0.1) ].describe().to_csv(save_fname.with_name(save_fname.stem + 'slope0.05-0.1_describe.csv'))
        logs_df[ (logs_df.mean_slope_gradient >= 0.1) ].describe().to_csv(save_fname.with_name(save_fname.stem + 'slope0.1-_describe.csv'))

        # denormalize images
        x_test = denorm_func_x(x_test)
        y_test = denorm_func_y(y_test)
        y_intp = denorm_func_x(y_intp)
        y_pred = denorm_func_y(y_pred)

        with open(save_fname.with_name('data_SR.pkl'), 'wb') as f:
            pickle.dump(y_pred, f)
        if save_all:
            with open(save_fname.with_name('data_BICUBIC.pkl'), 'wb') as f:
                pickle.dump(y_intp, f)
            with open(save_fname.with_name('data_HR.pkl'), 'wb') as f:
                pickle.dump(y_test, f)
            with open(save_fname.with_name('data_LR.pkl'), 'wb') as f:
                pickle.dump(x_test, f)

        if gen_plot:
            # plot a test image
            for i in tqdm(range(0, len(x_test), 4), desc='generate images'):
                Plotter.plot_test_imgs_2(
                    imgs=[x_test[i:i+4], y_intp[i:i+4], y_pred[i:i+4], y_test[i:i+4]],
                    labels=['low-res', 'bicubic', 'super-res', 'high-res'],
                    logs=logs[:,i:i+4,:],
                    img_indices=test_indices[i:i+4],
                    slope_grads=slopes[i:i+4],
                    save_fname=save_fname.with_name('test_result_{}-{}.png'.format(i, i+3)))
        
        return logs
        