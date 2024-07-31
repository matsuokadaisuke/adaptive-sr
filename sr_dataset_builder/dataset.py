import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

class Dataset:
    """
    Build a randomly-block-separated bathymetric chart dataset

    Attributes:
        x_train: low-resolution training data
        y_train: high-resolution training data
        x_validation: low-resolution validation data
        y_validation: high-resolution validation data
        x_test: low-resolution test data
        y_test: high-resolution test data
    """

    def __init__(self, lr_fname, hr_fname, block_shape=(3, 3)):
        """
        Args:
            lr_fname: low-resolution filename
            hr_fname: high-resolution filename
            block_shape: block shape
        """

        # load low-resolution image
        print('loading... ({})'.format(lr_fname), flush=True)
        with open(lr_fname, 'rb') as f:
            lr_img = pickle.load(f)
        # cleansing nan data
        lr_img = self.cleansing(lr_img)
        # splitting blocks
        #lr_blocks = self.block_split(lr_img, block_shape, img_shape=(32, 32), stride=(4, 4)) # if create 100m dataset
        lr_blocks = self.block_split(lr_img, block_shape, img_shape=(16, 16), stride=(2, 2)) # if create 200m dataset

        # load high-resolution image
        print('loading... ({})'.format(hr_fname), flush=True)
        with open(hr_fname, 'rb') as f:
            hr_img = pickle.load(f)
        # cleansing nan data
        hr_img = self.cleansing(hr_img)
        # splitting blocks
        hr_blocks = self.block_split(hr_img, block_shape, img_shape=(64, 64), stride=(8, 8)) # if create 50m dataset

        # expanding dimensions
        lr_blocks = np.expand_dims(lr_blocks, axis=-1)
        hr_blocks = np.expand_dims(hr_blocks, axis=-1)

        # remove images with nan
        print('removing nan...', end='')
        lr_blocks, hr_blocks = self.rm_nan(lr_blocks, hr_blocks)
        print(' done')
        # splitting train, validation and test data
        #print('splitting data...', end='')
        #x_train, x_validation, x_test, y_train, y_validation, y_test = self.train_validation_test_split(lr_blocks, hr_blocks)
        #print(' done')

        # pack images in blocks
        #self.x_train = np.concatenate(x_train, axis=0)
        #self.y_train = np.concatenate(y_train, axis=0)
        #self.x_validation = np.concatenate(x_validation, axis=0)
        #self.y_validation = np.concatenate(y_validation, axis=0)
        self.x_test = np.concatenate(lr_blocks, axis=0)
        self.y_test = np.concatenate(hr_blocks, axis=0)

    def cleansing(self, img, kernel_size=(5,5)):
        """
        Removing noises by mean of kernel (without nan)

        Args:
            img: input image
            kernel_size: averaging kernel size

        Returns:
            cleansed image
        """

        # fill zero pixel by nan
        fimg = np.copy(img)
        fimg = np.where(img == 0, np.nan, img)
        # get nan indices
        nan_indices = list(zip(*np.where(np.isnan(fimg))))

        # compute margins
        margin = np.array(kernel_size) // 2
        # cleansing the image
        cimg = np.copy(fimg)
        for i, j in tqdm(nan_indices, 'cleansing'):
            kernel = fimg[i - margin[0] : i + margin[0] + 1,
                     j - margin[1] : j + margin[1] + 1]
            if not np.isnan(kernel).all():
                cimg[i, j] = np.nanmean(kernel)

        return cimg

    def block_split(self, base_img, block_shape, img_shape, stride):
        """
        Split images in blocks

        Args:
            base_img: base image
            block_shape: block shape
            img_shape: trim image size
            stride: image sampling stride

        Returns:
            all images in blocks with shape (num_blocks, num_images, width, height)
        """

        # split original image to blocks with shape (num_blocks, block_w, block_h)
        blocks = np.concatenate(np.hsplit(
            np.stack(np.hsplit(base_img, block_shape[1])), block_shape[0]))
        # get all images in each blocks
        block_imgs = []
        for block in tqdm(blocks, 'splitting'):
            # get the number of offset in each axis
            offset = np.asarray(block.shape) - np.asarray(img_shape) + 1
            # slice images
            imgs = []
            for w in range(0, offset[0], stride[0]):
                for h in range(0, offset[1], stride[1]):
                    imgs.append(block[w:w + img_shape[0], h:h + img_shape[1]])
            block_imgs.append(imgs)

        return np.asarray(block_imgs)

    def rm_nan(self, lr_blocks, hr_blocks):
        """
        Remove images including nan

        Args:
            lr_blocks: low-resolution block images with shape (num_blocks, num_images, width, height)
            hr_blocks: high-resolution block images with shape (num_blocks, num_images, width, height)

        Returns:
            nan-removed low- and high-resolution images
        """

        # remove images with nan
        lr_block_imgs = []
        hr_block_imgs = []
        for lr_imgs, hr_imgs in zip(lr_blocks, hr_blocks):
            # false if images include at least one nan value
            indices = np.asarray([ not (np.isnan(lr_img).any() or np.isnan(hr_img).any())
                for lr_img, hr_img in zip(lr_imgs, hr_imgs) ])
            # get no-nan images
            if indices.any():
                lr_block_imgs.append(lr_imgs[indices])
                hr_block_imgs.append(hr_imgs[indices])

        return np.asarray(lr_block_imgs), np.asarray(hr_block_imgs)

    def train_validation_test_split(self, x, y, validation_size=0.1, test_size=0.1):
        """
        Split train, validation and test data

        Args:
            x: low-resolution block images with shape (num_blocks, num_images, width, height)
            y: high-resolution block images with shape (num_blocks, num_images, width, height)
            validation_size: validation data ratio (should be between 0.0 and 1.0)
            test_size: test data ratio (should be between 0.0 and 1.0)

        Returns:
            low-resolution train, validation and test data,
            high-resolution train, validation and test data
        """

        # check whether x- and y-lengths are the same
        if len(x) != len(y):
            raise ValueError('Unexpected number of images is detected.')

        # compute the number of train, validation and test images
        num_imgs = len(x)
        num_validation = int(num_imgs * validation_size)
        num_test = int(num_imgs * test_size)
        num_train = num_imgs - num_validation - num_test
        # split indices randomly
        indices = np.random.choice(num_imgs, num_imgs, replace=False)
        id_train = indices[0:num_train]
        id_validation = indices[num_train:num_train + num_validation]
        id_test = indices[num_train + num_validation:]

        return x[id_train], x[id_validation], x[id_test], \
            y[id_train], y[id_validation], y[id_test]

    def save(self, save_dir):
        """
        Save train, validation and test data

        Args:
            save_dir: save directory
        """
        print('saving ({})...'.format(save_dir), end='')

        # set output directories
        save_dir = Path(save_dir)
        train_dir = save_dir / 'train'
        validation_dir = save_dir / 'validation'
        test_dir = save_dir / 'test'
        # create directories
        dirs = [save_dir, train_dir, validation_dir, test_dir]
        for d in dirs:
            if not d.exists():
                d.mkdir()
        '''
        # save train data
        with open(train_dir / 'data_LR.pkl', 'wb') as f:
            pickle.dump(self.x_train, f)
        with open(train_dir / 'data_HR.pkl', 'wb') as f:
            pickle.dump(self.y_train, f)
        # save validation data
        with open(validation_dir / 'data_LR.pkl', 'wb') as f:
            pickle.dump(self.x_validation, f)
        with open(validation_dir / 'data_HR.pkl', 'wb') as f:
            pickle.dump(self.y_validation, f)
        '''
        # save test data
        with open(test_dir / 'data_LR.pkl', 'wb') as f:
            pickle.dump(self.x_test, f)
        with open(test_dir / 'data_HR.pkl', 'wb') as f:
            pickle.dump(self.y_test, f)
        '''
        # print train, validation and test shapes
        print(' done')
        print('----------------------------------------------------------------')
        print('     train: x = {}, y = {}'.format(
            self.x_train.shape, self.y_train.shape))
        print('validation: x = {}, y = {}'.format(
            self.x_validation.shape, self.y_validation.shape))
        print('      test: x = {}, y = {}'.format(
            self.x_test.shape, self.y_test.shape))
        print('----------------------------------------------------------------')
        '''