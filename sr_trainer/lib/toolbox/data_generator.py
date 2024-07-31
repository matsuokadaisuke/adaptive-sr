import pickle
import numpy as np
# import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import apply_affine_transform
import cv2


class DataGenerator:
    """
    Generate low- and high-resolution images

    Attributes:
        lr_imgs: low-resolution images
        hr_imgs: high-resolution images
        num_imgs: number of images
        batch_size: batch size
    """

    def __init__(self, x, y, norm_func, batch_size, da_conf=None):
        """
        Args:
            x: low-resolution images
            y: high-resolution images
            norm_func: normalization function
            batch_size: batch size
        """
        # remove images with nan
        indices = [ not (np.isnan(lr_img).any() or np.isnan(hr_img).any())
            for lr_img, hr_img in zip(x, y) ]
        self.lr_imgs = x[indices]
        self.hr_imgs = y[indices]
        # normalize images
        self.lr_imgs = norm_func(self.lr_imgs, np.nanmin(self.lr_imgs), np.nanmax(self.lr_imgs))
        self.hr_imgs = norm_func(self.hr_imgs, np.nanmin(self.hr_imgs), np.nanmax(self.hr_imgs))
        #self.lr_imgs = norm_func(self.lr_imgs)
        #self.hr_imgs = norm_func(self.hr_imgs)
        # set the number of images
        self.num_imgs = len(self.lr_imgs)
        # set batch size
        self.batch_size = batch_size
        # online data augmentation config
        self.online_da_valid = False
        self.da_conf = da_conf
        self.use_depth_scale = False
        self.depth_scale = [1.0, 1.0]
        self.zoom_iso = False
        self.zoom_range = 0
        self.horizontal_flip = False
        self.vertical_flip = False
        self.rotation_range = 0
        self.steps_per_epoch_mag = 1
        if self.da_conf is not None:
            self.online_da_valid = self.da_conf['valid']
            self.use_depth_scale = self.da_conf['use_depth_scale']
            self.depth_scale = self.da_conf['depth_scale_range']
            self.zoom_range = self.da_conf['zoom_range']
            self.zoom_iso = self.da_conf['zoom_iso']
            self.horizontal_flip = self.da_conf['horizontal_flip']
            self.vertical_flip = self.da_conf['vertical_flip']
            self.rotation_range = self.da_conf['rotation_range']

    def sample(self, num_samples, replace=False, indices=[]):
        """
        Sample low- and high-resolution images

        Args:
            num_samples: number of samples
            replace: duplicated selection is allowed for True
            indices: manually given indices

        Returns:
            sample images (low-resolution images, high-resolution images)
        """
        if len(indices) < 1:
            indices = np.arange(self.num_imgs)
            indices = np.random.choice(indices, num_samples, replace=replace)
        return self.lr_imgs[indices], self.hr_imgs[indices]

    def generator(self):
        """
        Generate low- and high-resolution images in batch size

        Yields:
            low-resolution and high-resolution images
        """

        while True:
            shuffled_indices = np.random.permutation(self.num_imgs)
            for i in range(self.steps_per_epoch()):

                yield self.sample(self.batch_size, 
                    indices=shuffled_indices[i*self.batch_size:(i+1)*self.batch_size])

    def random_depth_scale(self, img):
        # scale = 0.7
        # scale = self.depth_scale
        # zoom_range = [scale, 1 + (1-scale)]
        zoom_range = self.depth_scale
        zoom = np.random.rand() * (zoom_range[1]-zoom_range[0]) + zoom_range[0]
        mean = img.mean()
        img_ret = img - mean
        # print(zoom)
        img_ret = img_ret * zoom
        img_ret = img_ret + mean
        return img_ret

    def random_isotropic_zoom(self, img):
        ''' apply random zoom to x and y direction with same magnitude

        tf.keras.preprocessing.image.random_zoom zooms x and y independently
        '''
        zoom_range = self.zoom_range
        # if not isinstance(zoom_range,list) or not isinstance(zoom_range,tuple):
        if isinstance(zoom_range, int) or isinstance(zoom_range, float):
            zoom_range = [1-zoom_range, 1+zoom_range]
        zoom = np.random.rand() * (zoom_range[1]-zoom_range[0]) + zoom_range[0]
        img_ret = apply_affine_transform(img, zx=zoom, zy=zoom, fill_mode='reflect')
        return img_ret

    def pre_func(self, img):
        if self.use_depth_scale:
            img = self.random_depth_scale(img)
        if self.zoom_iso:
            img = self.random_isotropic_zoom(img)
        return img

    def generator_with_da(self, sr_scale=4):
        """
        Generate low- and high-resolution images in batch size
        - load high-resolution image
        - apply random scaling (zoom, depth) using keras ImageDataGenerator
        - generate low-resolution image by down sampling augmented high-resolution image

        Yields:
            low-resolution and high-resolution images
        """
        # zoom_range = 0
        # preprocessing_function = None
        # horizontal_flip = False
        # vertical_flip = False
        # rotation_range = 0
        # if self.da_conf is not None:
        #     self.depth_scale = self.da_conf['depth_scale_ratio']
        #     zoom_range = self.da_conf['zoom_range']
        #     self.zoom_range = zoom_range
        #     if self.da_conf['use_depth_scale']:
        #         preprocessing_function = self.random_depth_scale

        print('zoom_range', self.zoom_range)
        print('zoom_iso', self.zoom_iso)
        print('use_depth', self.use_depth_scale)
        print('depth_scale', self.depth_scale)

        img_gen = ImageDataGenerator(
            featurewise_center = False,#○
            samplewise_center = False,#○
            featurewise_std_normalization = False,#○
            samplewise_std_normalization = False,#○
            zca_whitening = False,
            zca_epsilon = 1e-06,
            rotation_range = self.rotation_range,#○
            width_shift_range = 0.0,#○
            height_shift_range = 0.0,#○
            # brightness_range = [0.5, 1.5],#○
            brightness_range = None,#○
            shear_range = 0.0,#○
            # zoom_range = [0.5,1.0],#○
            zoom_range = self.zoom_range if self.zoom_iso == False else 0,#○
            # zoom_range = 0.0,#○
            channel_shift_range = 0.0,#○
            fill_mode = "reflect",#○
            cval = 0.0,#○
            horizontal_flip = self.horizontal_flip,#○
            vertical_flip = self.vertical_flip,#○
            rescale = None,#○
            # preprocessing_function = None,#○
            # preprocessing_function = random_depth_scale,#○
            preprocessing_function = self.pre_func,#○
            data_format = None,
            validation_split = 0.0,
            dtype = None,
        )

        for hr_imgs in img_gen.flow(self.hr_imgs, batch_size=self.batch_size, shuffle=True):
            lr_imgs = np.array([cv2.resize(im, (hr_imgs.shape[2]//sr_scale, hr_imgs.shape[1]//sr_scale)).reshape(hr_imgs.shape[2]//sr_scale, hr_imgs.shape[1]//sr_scale, 1) for im in hr_imgs])
            yield (lr_imgs, hr_imgs)

    def steps_per_epoch(self):
        """
        Get the number of steps per an epoch (number of images / batch size)

        Returns:
            number of steps per an epoch
        """
        return self.num_imgs // self.batch_size
    
