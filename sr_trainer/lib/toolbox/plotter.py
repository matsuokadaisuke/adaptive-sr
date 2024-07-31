import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Plotter:
    """
    Plot history graphs and test images
    """

    @classmethod
    def __setup_ax(cls, ax, xlabel, ylabel,
        xscale_log=False, yscale_log=False, fontsize=9, labelsize=8):
        """
        Setup graph axis

        Args:
            ax: axis
            xlabel: x-axis label
            ylabel: y-axis label
            xscale_log: True set x-axis as log scale
            yscale_log: True set y-axis as log scale
            fontsize: font size
            labelsize: label size
        """

        # set log scales
        if xscale_log:
            ax.set_xscale('log')
        if yscale_log:
            ax.set_yscale('log')
        # set axis labels
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=fontsize)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=fontsize)
        # set tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        ax.tick_params(axis='both', which='minor', labelsize=labelsize)
        # set grid line colors
        ax.grid(which='major', color='darkgray', linewidth=0.5)
        ax.grid(which='minor', color='lightgray', linewidth=0.5)
        # set legend
        ax.legend(loc='upper left', fontsize=fontsize)

    @classmethod
    def __pair_plot(cls, ax, x, y, labels, color):
        """
        Plot training data and validation data

        Args:
            ax: plot axis
            x: epochs for x-axis
            y: training and validation data for y-axis
            labels: data labels
            color: plot color
        """
        ax.plot(x, y[0], label=labels[0],
            color=color, linewidth=1, marker=None, markersize=0, alpha=0.5)
        ax.plot(x, y[1], label=labels[1],
            color=color, linewidth=0, marker='o', markersize=1)

    @classmethod
    def plot_history_graphs(cls, epochs, logs, save_fname=None):
        """
        Plot history graph

        Args:
            epochs: epochs
            logs: logs including { 'loss':[], 'val_loss':[], 'psnr':[], 'val_psnr':[], 'dssim':[], 'val_dssim':[] }
            save_fname: save filename (If None, show is called)
        """

        # setup a figure
        fig, [ax_loss, ax_psnr, ax_dssim] = plt.subplots(nrows=1, ncols=3, figsize=(14, 3))
        # get colormap
        cmap = plt.get_cmap('tab10')

        # set a prefix for SRGAN
        is_gan = 'd_loss' in logs
        prefix = '' if not is_gan else 'g_'

        # plot loss
        loss_name = prefix + 'loss'
        cls.__pair_plot(ax=ax_loss, x=epochs, y=[logs[loss_name], logs['val_' + loss_name]], labels=[loss_name, 'val_' + loss_name], color=cmap(0))
        if is_gan:
            d_loss_name = 'd_' + 'loss'
            cls.__pair_plot(ax=ax_loss, x=epochs, y=[logs[d_loss_name], logs['val_' + d_loss_name]], labels=[d_loss_name, 'val_' + d_loss_name], color=cmap(3))
        cls.__setup_ax(ax_loss, xlabel='epoch', ylabel='loss', yscale_log=True)
        # plot psnr
        psnr_name = prefix + 'psnr'
        cls.__pair_plot(ax=ax_psnr, x=epochs, y=[logs[psnr_name], logs['val_' + psnr_name]], labels=['psnr', 'val_psnr'], color=cmap(1))
        cls.__setup_ax(ax_psnr, xlabel='epoch', ylabel='psnr', yscale_log=False)
        # plot dssim
        dssim_name = prefix + 'dssim'
        cls.__pair_plot(ax=ax_dssim, x=epochs, y=[logs[dssim_name], logs['val_' + dssim_name]], labels=['dssim', 'val_dssim'], color=cmap(2))
        cls.__setup_ax(ax_dssim, xlabel='epoch', ylabel='dssim', yscale_log=True)

        # adjust margins
        plt.subplots_adjust(wspace=0.25, hspace=0.0,
            left=0.075, right=0.975, bottom=0.2, top=0.9)
        # save the figure
        if save_fname == None:
            plt.show()
        else:
            plt.savefig(save_fname)
        # close the figure
        plt.cla()
        plt.clf()
        plt.close('all')
        # explicitly free memory
        gc.collect()

    @classmethod
    def plot_test_imgs(cls, imgs, labels, logs, save_fname=None,
        figsize=(8, 8), fontsize=9, labelsize=8, textsize=7):
        """
        Plot test images

        Args:
            imgs: plot images with shape (num_groups, num_samples, width, height, channel)
            labels: group names
            logs: logs with shape (num_groups, num_samples, psnr and dssim)
            save_fname: save filename (If None, show is called)
            figsize: figure size (x 100px)
            fontsize: title font size
            labelsize: tick font size
            textsize: text size
        """

        # get the number of columns (sample images)
        nrows = len(imgs[-1])
        # get the number of rows (groups of input, bicubic, sr, true)
        ncols = len(imgs)

        # create a figure
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        # get colormap
        cmap = plt.get_cmap('tab10')
        # set text box properties
        bbox = {
            'facecolor' : 'white',
            'edgecolor' : 'lightgray',
            'boxstyle'  : 'round',
            'linewidth' : 1,
            'alpha'     : 0.5,
        }

        # loop for samples
        for i in range(nrows):
            # get colorbar range from true images
            vmin, vmax = np.min(imgs[-1][i]), np.max(imgs[-1][i])
            # loop for input, bicubic, sr and true images
            for j, (img, label) in enumerate(zip(imgs, labels)):
                # plot an image
                mappable = ax[i, j].imshow(img[i].squeeze(), cmap='jet',
                    norm=Normalize(vmin=vmin, vmax=vmax), interpolation='none')
                # draw psnr and dssim texts
                psnr, dssim, _ = logs[j, i]
                if not np.isnan(psnr) and not np.isnan(dssim):
                    ax[i, j].text(0.1, 0.1, 'psnr: {:3.1f}\ndssim: {:3.1e}'.format(psnr, dssim),
                        color='k', fontsize=textsize, bbox=bbox, transform=ax[i,j].transAxes)

                # set title
                ax[i, j].set_title(label,
                    fontweight='bold', fontsize=fontsize, color=cmap(j))
                # set tick label size
                ax[i, j].tick_params(labelsize=labelsize)
                # set x- and y-ticks
                ax[i, j].set_xticks([0, img.shape[1] // 2, img.shape[1] - 1])
                ax[i, j].set_yticks([0, img.shape[2] // 2, img.shape[2] - 1])
                # plot colorbars
                divider = make_axes_locatable(ax[i, j])
                cax = divider.append_axes("right", size="10%", pad=0.1)
                cb = fig.colorbar(mappable, cax=cax)
                #cb.set_clim(vmin, vmax)
                cb.mappable.set_clim(vmin, vmax)
                cb.ax.tick_params(labelsize=labelsize)
                # remove colorbars except for right edge
                if j < ncols - 1:
                    cb.remove()

        # adjust margins
        plt.subplots_adjust(wspace=0.1, hspace=0.0,
            left=0.1, right=0.9, bottom=0.01, top=0.99)
        # save the figure
        if save_fname == None:
            plt.show()
        else:
            plt.savefig(save_fname)
        # close the figure
        plt.cla()
        plt.clf()
        plt.close('all')
        # explicitly free memory
        gc.collect()


    @classmethod
    def plot_test_imgs_2(cls, imgs, labels, logs, img_indices, slope_grads,
        save_fname=None,
        figsize=(8, 8), fontsize=9, labelsize=8, textsize=7,
        ):
        """
        Plot test images2.

        Args:
            imgs: plot images with shape (num_groups, num_samples, width, height, channel)
            labels: group names
            logs: logs with shape (num_groups, num_samples, psnr and dssim)
            img_indices: indices of images
            slope_grads: list of mean slope gradient of images
            save_fname: save filename (If None, show is called)
            figsize: figure size (x 100px)
            fontsize: title font size
            labelsize: tick font size
            textsize: text size
        """

        # get the number of columns (sample images)
        nrows = len(imgs[-1])
        # get the number of rows (groups of input, bicubic, sr, true)
        ncols = len(imgs)

        # create a figure
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        # get colormap
        cmap = plt.get_cmap('tab10')
        # set text box properties
        bbox = {
            'facecolor' : 'white',
            'edgecolor' : 'lightgray',
            'boxstyle'  : 'round',
            'linewidth' : 1,
            'alpha'     : 0.5,
        }

        # loop for samples
        for i in range(nrows):
            index = img_indices[i]
            slope = slope_grads[i]
            # get colorbar range from true images
            vmin, vmax = np.min(imgs[-1][i]), np.max(imgs[-1][i])
            # loop for input, bicubic, sr and true images
            for j, (img, label) in enumerate(zip(imgs, labels)):
                # plot an image
                mappable = ax[i, j].imshow(img[i].squeeze(), cmap='jet',
                    norm=Normalize(vmin=vmin, vmax=vmax), interpolation='none')
                # draw psnr and dssim texts
                psnr, dssim, _ = logs[j, i]
                if not np.isnan(psnr) and not np.isnan(dssim):
                    title = label + '_{}\npsnr:{:3.1f}, dssim:{:3.1e}'.format(index, psnr, dssim)
                else:
                    title = label + '_{}\nmean_slope_grad:{:.4f}'.format(index, slope)
                  
                # set title
                ax[i, j].set_title(title, fontsize=8)
             
                ax[i, j].axes.xaxis.set_visible(False)
                ax[i, j].axes.yaxis.set_visible(False)
                # plot colorbars
                divider = make_axes_locatable(ax[i, j])
                cax = divider.append_axes("right", size="10%", pad=0.1)
                cb = fig.colorbar(mappable, cax=cax)
                #cb.set_clim(vmin, vmax)
                cb.mappable.set_clim(vmin, vmax)
                cb.ax.tick_params(labelsize=labelsize)
                # remove colorbars except for right edge
                if j < ncols - 1:
                    cb.remove()

        # adjust margins
        plt.subplots_adjust(wspace=0.1, hspace=0.0,
            left=0.1, right=0.9, bottom=0.01, top=0.99)
        # save the figure
        if save_fname == None:
            plt.show()
        else:
            plt.savefig(save_fname)
        # close the figure
        plt.cla()
        plt.clf()
        plt.close('all')
        # explicitly free memory
        gc.collect()