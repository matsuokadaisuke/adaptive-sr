import pickle
import matplotlib.pyplot as plt
import numpy as np 
import os
import argparse
import yaml
import tqdm
from tensorflow.keras.preprocessing.image import apply_affine_transform

def depth_scale(img, scale):
    # scale = 0.7
    mean = img.mean()
    img_ret = img - mean
    img_ret = img_ret * scale
    img_ret = img_ret + mean
    return img_ret

def isotropic_zoom(img, zoom):
    ''' apply zoom to x and y direction with same magnitude

    tf.keras.preprocessing.image.random_zoom zooms x and y independently
    '''
    
    img_ret = apply_affine_transform(img, zx=zoom, zy=zoom, fill_mode='reflect')
    return img_ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('yml_fname',help='filename of input parameters (.yml)')
    args = parser.parse_args()
    fname = args.yml_fname
    with open(fname, 'rb') as f:
        param_dict = yaml.full_load(f)
    print(param_dict)
    train_dir = param_dict['train_dir']
    train_datas = param_dict['train_data']
    train_savedatas = param_dict['train_savedata']
    dscales = param_dict['dscales']
    xyscales = param_dict['xyscales']

    for train_data,train_savedata in zip(train_datas,train_savedatas):
        print(train_data)
        with open(os.path.join(train_dir,train_data), mode='rb') as f:
            imgs_org = pickle.load(f)
        imgs = imgs_org.copy()
        imgs = imgs.tolist()
        print(type(imgs))
        for i in tqdm.tqdm(range(len(imgs_org))):
            org_img = imgs_org[i]
            for ds in dscales:
                img_ds = depth_scale(org_img, ds)
                img_ds = img_ds[np.newaxis,:,:]
                imgs.extend(img_ds)
            for z in xyscales:
                img_z = isotropic_zoom(org_img, z)
                img_z = img_z[np.newaxis,:,:]
                imgs.extend(img_z)

        imgs = np.array(imgs)
        print(imgs.shape)
        with open(os.path.join(train_dir,train_savedata), mode='wb') as f:
            pickle.dump(imgs, f, protocol=4)
            print('save: '+ str(os.path.join(train_dir,train_savedata)))
