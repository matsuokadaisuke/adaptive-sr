import pickle
import cv2
import numpy as np
# from dataset import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('oki_test/oki_hr_zz_test.pkl', mode='rb') as f:
        img = pickle.load(f)

    plt.imshow(img)
    plt.show()

    fimg = np.copy(img)
    fimg = np.where(img == 0, np.nan, img)
    nan_indices = list(zip(*np.where(np.isnan(fimg))))
    kernel_size=(5,5)
    margin = np.array(kernel_size) // 2
    cimg = np.copy(fimg)
    for i, j in tqdm(nan_indices, 'cleansing'):
        kernel = fimg[i - margin[0] : i + margin[0] + 1,
                    j - margin[1] : j + margin[1] + 1]
        if not np.isnan(kernel).all():
            cimg[i, j] = np.nanmean(kernel)


    plt.imshow(cimg)
    plt.show()
    print(cimg.shape)

    dimg = np.nan_to_num(cimg)
    print(dimg)
    img_resize = cv2.resize(dimg, (dimg.shape[1]//4, dimg.shape[0]//4))

    # print(img_resize)

    with open('oki_test/oki_lr_zz_test.pkl', mode="wb") as f:
        pickle.dump(img_resize, f)

    plt.imshow(img_resize)
    plt.show()
