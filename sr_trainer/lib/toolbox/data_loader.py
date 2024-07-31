import pickle
import numpy as np
from pathlib import Path
import cv2
import pandas as pd


def normalize_alt(data):
    #norm_range = [-2500, 0]
    norm_range = [np.nanmin(data), np.nanmax(data)]
    data_norm = (data - norm_range[0]) / (norm_range[1] - norm_range[0])
    return data_norm, norm_range
class DataLoader:
    """
    Load train, validation and test data

    Attributes:
        dataset: train, validation and test data dictionary
    """

    def __init__(self, data_dir, sub_dirs=['train', 'validation', 'test'], sr_scale=None, normalize=False):
        """
        Args:
            data_dir: target dataset directory
            sub_dirs: training, validation and test sub-directories
        """
        self.sr_scale = sr_scale
        # load images
        self.dataset = {}
        self.features = {}
        for dir_name in sub_dirs:
            # set target directory
            dir = Path(data_dir) / dir_name
            # find pkl files
            fpaths = [ fpath for fpath in dir.glob('*.pkl') if fpath.is_file() ]
            # check the number of pkl files
            if len(fpaths) > 2:
                raise FileExistsError(
                    "Unnecessary file is found in '{}'".format(dir_name))
            # load images
            imgs = []
            feature_csv = None
            for fpath in fpaths:
                    with open(fpath, 'rb') as f:
                        imgs.append(np.asarray(pickle.load(f)))
                    if fpath.with_suffix('.csv').exists():
                        feature_csv = pd.read_csv(fpath.with_suffix('.csv'))
            x, y = imgs
            # swap x (low-resolution) and y (high-resolution)
            if x.shape[1] > y.shape[1]:
                x, y = y, x
            
            if sr_scale is not None:
                # create low-resolution images by downsampling high-resolution. do not use original LR images
                x = np.array([cv2.resize(im, (y.shape[2]//sr_scale, y.shape[1]//sr_scale)).reshape(y.shape[2]//sr_scale, y.shape[1]//sr_scale, 1) for im in y])
            # store images in the dictionary
            if normalize:
                x, self.x_norm_range = normalize_alt(x)
                y, self.y_norm_range = normalize_alt(y)
                
            self.dataset[dir_name] = x, y
            self.features[dir_name] = feature_csv

    def train(self):
        """
        Get training dataset

        Returns:
            training dataset (x, y) with shape (num_imgs, width, height, channel)
        """
        return self.dataset['train']

    def validation(self):
        """
        Get validation dataset

        Returns:
            validation dataset (x, y) with shape (num_imgs, width, height, channel)
        """
        return self.dataset['validation']

    def test(self):
        """
        Get test dataset

        Returns:
            test dataset (x, y) with shape (num_imgs, width, height, channel)
        """
        return self.dataset['test']

    def sample_with_slope_range(self, dir_name, slmin, slmax):
        """Generate dataset with slope gradient range

        use slope gradient calculated with normalized(-2500,0) bathymetric data
        Args:
            dir_name (str): train, test or validation
            slmin (float): slope gradient minimum
            slmax (float): slope gradient max

        Returns:
            tuple(np.array, np.array): lr_images, hr_images with specified slope range
        """
        data = self.dataset[dir_name]
        f = self.features[dir_name]
        if f is None:
            'slope information csv not found!'
            return 
        key = 'mean_slope_gradient_norm_data'
        #key = 'normalized_mean_slope_gradient'
        inds =  f[(f[key] >= slmin) & (f[key] < slmax)].index
        return data[0][inds], data[1][inds], inds, f.loc[inds][key].values, self.x_norm_range, self.y_norm_range
    
    def return_norm_range(self):
        return self.x_norm_range, self.y_norm_range


def get_slope(h, delta=50):
    nx = 64
    ny = 64
    dx = delta
    dy = delta

    s = np.zeros((nx, ny))

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            sx = (h[i-1, j+1]+h[i-1, j]+h[i-1, j-1])-(h[i+1,j+1]+h[i+1, j]+h[i+1, j-1])
            sx = sx/(6.0*dx)
            sy = (h[i-1, j-1]+h[i, j-1]+h[i+1, j-1])-(h[i-1, j+1]+h[i,j+1]+h[i+1, j+1])
            xy = sy/(6.0*dy)
            s[i,j] = np.sqrt(sx*sx+sy*sy)
    return np.mean(s)


if __name__ == '__main__':

    # for checking
    loader =  DataLoader(r'D:\work\murai\dev\jamstec\sr\sr_trainer\data\output_origin',sub_dirs=['test'], sr_scale=4, normalize=True)
   
    x, y, inds = loader.sample_with_slope_range('test', 0.01, 0.1)
    for img, index in zip(y[:100], inds[:100]):
        print(index, get_slope(img, 50./2500))