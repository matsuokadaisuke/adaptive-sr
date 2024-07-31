import pickle
import matplotlib.pyplot as plt
import numpy as np 
import os
import argparse
import yaml
import tqdm
from tensorflow.keras.preprocessing.image import apply_affine_transform

SHOW = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('yml_fname',help='filename of input parameters (.yml)')
    args = parser.parse_args()
    fname = args.yml_fname
    with open(fname, 'r') as f:
        param_dict = yaml.full_load(f)
    print(param_dict)
    train_dir = param_dict['train_dir']
    train_datas = param_dict['train_data']
    train_savedatas = param_dict['train_savedata']
    fill_mode = param_dict['fill_mode']
    angles = param_dict['angles']

    for train_data,train_savedata in zip(train_datas,train_savedatas):
        print(train_data)        
        with open(os.path.join(train_dir,train_data), mode='rb') as f:
            imgs_org = pickle.load(f)
        print(imgs_org.shape)
        imgs = imgs_org.copy()
        imgs = imgs.tolist()
        for i in tqdm.tqdm(range(len(imgs_org))):
            if SHOW:
                plt.imshow(imgs_org[i][:,:,0], cmap='jet')
                plt.show()
            for a in angles:
                # img_rot = apply_affine_transform(hoge[i][:,:,0], theta=a, fill_mode=FILL_MODE)
                img_rot = apply_affine_transform(imgs_org[i], theta=a, fill_mode=fill_mode)
                if SHOW:
                    plt.imshow(img_rot[:,:,0], cmap='jet')
                    plt.show()
                # img_rot = img_rot[np.newaxis,:,:,np.newaxis]
                img_rot = img_rot[np.newaxis,:,:]
                imgs.extend(img_rot)
            
        imgs = np.array(imgs)
        print(imgs.shape)
        with open(os.path.join(train_dir,train_savedata), mode='wb') as f:
            pickle.dump(imgs, f, protocol=4)
            print('save: '+ str(os.path.join(train_dir,train_savedata)))
