import pickle
import matplotlib.pyplot as plt
import numpy as np 
import os
import argparse
import yaml
import tqdm

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
    train_savedir = param_dict['train_savedir']

    for train_data in train_datas:
        print(train_data)
        with open(os.path.join(train_dir,train_data), mode='rb') as f:
            imgs_org = pickle.load(f)
        #print(hoge.shape)
        imgs = imgs_org.copy()
        imgs = imgs.tolist()
        print(type(imgs))
        for i in tqdm.tqdm(range(len(imgs_org))):
            img_ud = np.flipud(imgs_org[i][:,:,0]) # 上下反転
            img_lr = np.fliplr(imgs_org[i][:,:,0]) # 左右反転
            img_udlr = np.flip(imgs_org[i][:,:,0], (0, 1)) # 上下左右反転

            img_ud = img_ud[np.newaxis,:,:,np.newaxis]
            img_lr = img_lr[np.newaxis,:,:,np.newaxis]
            img_udlr = img_udlr[np.newaxis,:,:,np.newaxis]
            # print(hoge_ar90.shape) # (1, 64, 64, 1)
            imgs.extend(img_ud.tolist())
            imgs.extend(img_lr.tolist())
            imgs.extend(img_udlr.tolist())

        imgs = np.array(imgs)
        print(imgs.shape)
        os.makedirs(train_savedir, exist_ok=True)
        with open(os.path.join(train_savedir, train_data), mode='wb') as f:
            pickle.dump(imgs, f, protocol=4)
            print('save: '+ str(os.path.join(train_savedir, train_data)))
