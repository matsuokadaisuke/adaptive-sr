import numpy as np
import pickle
import pandas as pd
import argparse
import yaml
import os, sys
import subprocess
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from natsort import natsorted
from natsort import index_natsorted
import random
import re
import shutil
import warnings
warnings.simplefilter('error', category=RuntimeWarning)

def load_csv_norm(data_csv, target_key, target_data_csv, save_dir, original_image_num):
    df = pd.read_csv(data_csv)
    
    df_org = [df.loc[df['index'] < original_image_num, :]]
    df_flip = df.loc[(df['index'] >= original_image_num) & (df['index'] < original_image_num * 4), :]
    df_rotation = df.loc[df['index'] >= original_image_num * 4, :]
    
    split_num = len(df_flip) // original_image_num
    df_org.extend([df_flip.loc[df_flip['index'] % split_num == m, :] for m in range(split_num)])
    
    split_num = len(df_rotation) // original_image_num
    df_org.extend([df_rotation.loc[df_rotation['index'] % split_num == m, :] for m in range(split_num)])
    
    target_df = pd.read_csv(target_data_csv)
    ret = np.histogram(target_df[target_key], bins=20)
    ind = list(ret[0]).index(max(ret[0]))
    th = (ret[1][ind] + ret[1][ind + 1]) / 2
    
    slope_high, slope_low = [[] for m in range(len(df_org))], [[] for m in range(len(df_org))]

    index_high, index_low = [[] for m in range(len(df_org))], [[] for m in range(len(df_org))]
    
    for alt, num in zip(df_org[0][target_key], df_org[0]['index']):
        if alt >= th:
            slope_high[0].append(alt)
            index_high[0].append(num)
        else:
            slope_low[0].append(alt)
            index_low[0].append(num)
            
    ret = np.histogram(slope_high[0], bins=10)
    norm_ret = ret[0]
    norm_ret.sort()
    
    ret_low = np.histogram(slope_low[0], bins=10)
    
    for i in range(len(norm_ret)):
        count = 0
        for grad, ind in zip(slope_low[0], index_low[0]):
            if count == norm_ret[i]:
                break
            
            if grad >= ret_low[1][i] and grad < ret_low[1][i+1]:
                count += 1
                slope_high[0].append(grad)
                index_high[0].append(ind)
                
    slope_low[0] = []
    index_low[0] = []

    for alt, num in zip(df_org[0][target_key], df_org[0]['index']):
        if num not in index_high[0] and num not in index_low[0]:
            slope_low[0].append(alt)
            index_low[0].append(num)
    
    for i in range(1, len(df_org)):
        for j in index_high[0]:
            slope_high[i].append(df_org[i][target_key].iloc[j])
            index_high[i].append(df_org[i]['index'].iloc[j])
        
        for j in index_low[0]:
            slope_low[i].append(df_org[i][target_key].iloc[j])
            index_low[i].append(df_org[i]['index'].iloc[j])
    
    os.makedirs(f'{save_dir}/split_data', exist_ok=True)
    
    for i in range(len(slope_high)):
        plt.figure()
        plt.hist(slope_high[i], bins=20)
        plt.xlim(0, 1)
        plt.savefig(f'{save_dir}/split_data/split_data_high_{i}.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure()
        plt.hist(slope_low[i], bins=20)
        plt.xlim(0, 1)
        plt.savefig(f'{save_dir}/split_data/split_data_low_{i}.jpg', dpi=300, bbox_inches='tight')
        plt.close()
    
    for i in range(len(df_org)):
        slope_sort_index = index_natsorted(slope_high[i], reverse=True)
        slope_high[i] = [slope_high[i][m] for m in slope_sort_index]
        index_high[i] = [index_high[i][m] for m in slope_sort_index]

        slope_sort_index = index_natsorted(slope_low[i])
        slope_low[i] = [slope_low[i][m] for m in slope_sort_index]
        index_low[i] = [index_low[i][m] for m in slope_sort_index]
    
    return index_high, index_low, slope_high, slope_low, df, target_df
    
def load_imgs(imgs_file):
    with open(imgs_file, 'rb') as f:
        imgs_org = pickle.load(f)
    
    imgs = []

    for img in tqdm(imgs_org):
        imgs.append(list(img.reshape(img.shape[0], img.shape[1])))
    imgs = np.array(imgs)
    
    print(imgs.shape)
    
    return imgs, imgs_org

def mixup_norm(imgs_HR, imgs_HR_org, imgs_LR, imgs_LR_org, index_high, index_low, train_dir, target_df, target_key, slope_normalized):
    imgs_HR_low = [[] for m in range(len(index_low))]
    imgs_HR_high = [[] for m in range(len(index_high))]
    imgs_LR_low = [[] for m in range(len(index_low))]
    imgs_LR_high = [[] for m in range(len(index_high))]
    
    for i, imgs_index_low in enumerate(index_low):
        imgs_HR_low[i] = imgs_HR[imgs_index_low]
        imgs_LR_low[i] = imgs_LR[imgs_index_low]
    
    imgs_HR_low = np.array(imgs_HR_low)
    imgs_LR_low = np.array(imgs_LR_low)
    
    for i, imgs_index_high in enumerate(index_high):
        imgs_HR_high[i] = imgs_HR[imgs_index_high]
        imgs_LR_high[i] = imgs_LR[imgs_index_high]
    
    imgs_HR_high = np.array(imgs_HR_high)
    imgs_LR_high = np.array(imgs_LR_high)
    
    df_org = pd.read_csv(os.path.join(train_dir, '../data_LR.csv'))
    
    ret = np.histogram(df_org[target_key], bins=20)
    ret_target = np.histogram(target_df[target_key], bins=20)
    
    ratio = 10
    change_num = ret_target[0] * ratio - ret[0]
    change_num = np.where(change_num < 0, 0, change_num)
    size = sum(change_num)
    
    norm_range = [np.nanmin(imgs_LR_org), np.nanmax(imgs_LR_org)]
    mixup_HR = []
    mixup_LR = []
    count_l = 0
    count_h = 0
    weights_temp = np.random.beta(0.5, 0.5, 50)
    u_weights = np.unique(weights_temp)
    weights = natsorted(u_weights)
    mixup_data = []
    
    count_imgs = 0
    kind_num = len(index_high)
    df_mixup_slope = pd.read_csv('mixup_slope.csv')
    df_g = df_mixup_slope.groupby('High').groups
    slope_high_list = list(df_g.keys())
    mixup_pairs = []
    max_pair = len(imgs_LR_high) * len(imgs_LR_high[0]) * len(imgs_LR_low) * len(imgs_LR_low[0])
    for i in range(len(change_num)):
        print(f'mixup_range: {ret_target[1][i].round(3)}-{ret_target[1][i+1].round(3)}, mixup_num: {change_num[i]}')
        for j in tqdm(range(change_num[i])):
            if len(mixup_pairs) ==  max_pair:
                continue
            set_num_h = np.arange(kind_num)
            while(1):
                kind_h = set_num_h[random.randint(0, kind_num-1)]
                set_num_l = set_num_h[set_num_h != kind_h]
                kind_l = set_num_l[random.randint(0, kind_num-2)]
                mixup_pair = [kind_h, count_h, kind_l, count_l]
                if mixup_pair not in mixup_pairs:
                    mixup_pairs.append(mixup_pair)
                    break
                
            base_slope_high = get_slope_LR(imgs_LR_high[kind_h][count_h], delta=200/abs(np.diff(norm_range)[0]))
            base_slope_low = get_slope_LR(imgs_LR_low[kind_h][count_h], delta=200/abs(np.diff(norm_range)[0]))
            target_slope = (ret_target[1][i] + ret_target[1][i + 1]) / 2
            
            slope_high_id = idx_of_the_nearest(slope_high_list, base_slope_high)
            slope_ind = list(df_g[list(df_g.keys())[slope_high_id]])
            slope_low_list = list(df_mixup_slope['Low'][slope_ind])
            
            slope_low_id = indices_of_the_nearest(slope_low_list, base_slope_low)
            weight = np.array(df_mixup_slope['Weight'][slope_ind])[slope_low_id]
            mixup_slope = np.array(df_mixup_slope['slope'][slope_ind])[slope_low_id]
            
            weight = weight[idx_of_the_nearest(mixup_slope, target_slope)]
            img = imgs_LR_high[kind_h][count_h] * weight + imgs_LR_low[kind_l][count_l] * (1 - weight)
            mixup_LR.append(img)
            
            img = imgs_HR_high[kind_h][count_h] * weight + imgs_HR_low[kind_l][count_l] * (1 - weight)
            mixup_HR.append(img)
            mixup_data.append([count_imgs, kind_h, count_h, kind_l, count_l, weights, index_high[kind_h][count_h], index_low[kind_l][count_l]])
            
            count_imgs += 1
            count_l += 1
            count_h += 1
            if count_h == len(index_high[0]):
                count_h = 0
                
            if count_l == len(index_low[0]):
                count_l = 0

    mixup_HR_trans = []
    
    for img in mixup_HR:
        mixup_HR_trans.append(np.array([img]).T)

    mixup_HR_trans = np.array(mixup_HR_trans)
    imgs_HR_org_mixup = np.append(imgs_HR_org, mixup_HR_trans, axis=0)
            
    mixup_LR_trans = []
    for img in mixup_LR:
        mixup_LR_trans.append(np.array([img]).T)

    mixup_LR_trans = np.array(mixup_LR_trans)
    imgs_LR_org_mixup = np.append(imgs_LR_org, mixup_LR_trans, axis=0)
    
    return imgs_HR_org_mixup, mixup_HR_trans, mixup_HR, imgs_HR_high, imgs_HR_low, imgs_LR_org_mixup, mixup_LR_trans, mixup_LR, imgs_LR_high, imgs_LR_low, mixup_data

def get_slope(h, delta=50):
    nx = 64
    ny = 64
    dx = delta
    dy = delta

    s = np.zeros((nx, ny))

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            sx = (h[i-1, j+1]+h[i-1, j]+h[i-1, j-1]) - \
                (h[i+1, j+1]+h[i+1, j]+h[i+1, j-1])
            sx = sx/(6.0*dx)
            sy = (h[i-1, j-1]+h[i, j-1]+h[i+1, j-1]) - \
                (h[i-1, j+1]+h[i, j+1]+h[i+1, j+1])
            #xy = sy/(6.0*dy)
            sy = sy/(6.0*dy)
            s[i, j] = np.sqrt(sx*sx+sy*sy)
    return np.mean(s)

def get_slope_LR(h, delta):
    nx = 16
    ny = 16
    dx = delta
    dy = delta

    s = np.zeros((nx, ny))

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            sx = (h[i-1, j+1]+h[i-1, j]+h[i-1, j-1]) - \
                (h[i+1, j+1]+h[i+1, j]+h[i+1, j-1])
            sx = sx/(6.0*dx)
            sy = (h[i-1, j-1]+h[i, j-1]+h[i+1, j-1]) - \
                (h[i-1, j+1]+h[i, j+1]+h[i+1, j+1])
            #xy = sy/(6.0*dy)
            sy = sy/(6.0*dy)
            s[i, j] = np.sqrt(sx*sx+sy*sy)
    return np.mean(s)

def pick_images(target_df, save_dir, imgs_HR_mixup, imgs_LR_mixup, original_image_num, target_key, data_csv, increase_rate):
    df_mixup = pd.read_csv(os.path.join(save_dir, 'data_LR.csv'))
    ret_o_fix, ret_o = np.histogram(target_df[target_key], bins=20)
    
    df = pd.read_csv(data_csv)
    new_norm = [round(increase_rate * m) for m in ret_o_fix]
    
    slice_num = len(df) // original_image_num
    
    df_fix = []
    
    tmp = [df_mixup.loc[(df_mixup[target_key] >= ret_o[m]) & (df_mixup[target_key] <= ret_o[m + 1]), :] for m in range(len(new_norm))]
    
    tmp_mixup_num = [len(tmp[m].loc[(tmp[m]['index'] < original_image_num) | (tmp[m]['index'] >= original_image_num * slice_num), :]) for m in range(len(tmp))]
    
    mixup_use_num = []
    diff_num = []

    for i in range(len(new_norm)):
        if tmp_mixup_num[i] >= new_norm[i]:
            mixup_use_num.append(new_norm[i])
            diff_num.append(new_norm[i] - tmp_mixup_num[i])
        else:
            mixup_use_num.append(tmp_mixup_num[i])
            diff_num.append(new_norm[i] - tmp_mixup_num[i])
            
    tmp_flip_num = [len(tmp[m].loc[(tmp[m]['index'] >= original_image_num) & (tmp[m]['index'] < original_image_num * slice_num), :]) for m in range(len(tmp))]
    
    flip_use_num = []
    
    for i in range(len(new_norm)):
        if diff_num[i] > 0:
            if tmp_flip_num[i] >= diff_num[i]:
                flip_use_num.append(diff_num[i])
                diff_num[i] = 0
            else:
                flip_use_num.append(tmp_flip_num[i])
                diff_num[i] -= tmp_flip_num[i]
                
        else:
            flip_use_num.append(0)
                
    for i in range(len(diff_num)):
        ind = []
        if diff_num[i] > 0:
            for j in range(len(diff_num)):
                if diff_num[j] < 0:
                    ind.append(abs(diff_num.index(diff_num[i]) - diff_num.index(diff_num[j])))
                else:
                    ind.append(100)
                    
            ind = index_natsorted(ind)
            
            for j in ind:
                if abs(diff_num[j]) > diff_num[i]:
                    mixup_use_num[j] += diff_num[i]
                    diff_num[j] += diff_num[i]
                    diff_num[i] = 0
                    break
                
                else:
                    mixup_use_num[j] += diff_num[j]
                    diff_num[i] -= diff_num[j]
                      
    for i in range(len(mixup_use_num)):
        mnum_imgs = mixup_use_num[i]
        fnum_imgs = flip_use_num[i]

        area = [ret_o[i], ret_o[i + 1]]
        tmp = df_mixup.loc[(df_mixup[target_key] >= area[0]) & (df_mixup[target_key] <= area[1]), :]
        
        tmp_mixup = tmp.loc[(tmp['index'] < original_image_num) | (tmp['index'] >= original_image_num * slice_num), :]
        tmp_mixup = tmp_mixup.sort_values('mean_altitude')
        
        tmp_flip = tmp.loc[(tmp['index'] >= original_image_num) & (tmp['index'] < original_image_num * slice_num), :]
        tmp_flip = tmp_flip.sort_values('mean_altitude')
        
        if mnum_imgs != 0:
            step = len(tmp_mixup) // mnum_imgs
            count = 0
            if step == 0:
                step = 1

            for i, (index, row) in enumerate(tmp_mixup.iterrows()):
                if count == mnum_imgs:
                    break
                
                if i % step == 0:
                    df_fix.append({'index': row['index'], 'mean_slope_gradient': row['mean_slope_gradient'], 'altitude_difference': row['altitude_difference'], 'mean_altitude': row['mean_altitude'],
                            'mean_slope_gradient_norm_data': row['mean_slope_gradient_norm_data'], 'normalized_mean_slope_gradient': row['normalized_mean_slope_gradient']})
                    count += 1
        
        if fnum_imgs != 0:    
            step = len(tmp_flip) // fnum_imgs
            count = 0
            if step == 0:
                step = 1
            
            for i, (index, row) in enumerate(tmp_flip.iterrows()):
                if count == fnum_imgs:
                    break
                
                if i % step == 0:
                    df_fix.append({'index': row['index'], 'mean_slope_gradient': row['mean_slope_gradient'], 'altitude_difference': row['altitude_difference'], 'mean_altitude': row['mean_altitude'],
                            'mean_slope_gradient_norm_data': row['mean_slope_gradient_norm_data'], 'normalized_mean_slope_gradient': row['normalized_mean_slope_gradient']})
                    count += 1
                    
    df_fix = pd.DataFrame(df_fix)

    imgs_HR_fix, imgs_LR_fix = [], []
    for ind in tqdm(df_fix['index']):
        imgs_HR_fix.append(imgs_HR_mixup[int(ind)])
        imgs_LR_fix.append(imgs_LR_mixup[int(ind)])
        
    imgs_HR_fix = np.array(imgs_HR_fix)
    imgs_LR_fix = np.array(imgs_LR_fix)
    
    picked_img_save_dir = os.path.join(save_dir, 'picked_images')
    os.makedirs(picked_img_save_dir, exist_ok=True)
    
    with open(os.path.join(picked_img_save_dir, train_datas[0]), 'wb') as f:
        pickle.dump(imgs_HR_fix, f)
    with open(os.path.join(picked_img_save_dir, train_datas[1]), 'wb') as f:
        pickle.dump(imgs_LR_fix, f)
        
    subprocess.run(['python', 'bathymetric_map_feature_LR.py', os.path.join(picked_img_save_dir, 'data_LR.pkl')])

def idx_of_the_nearest(data, value):
    idx = np.argmin(np.abs(np.array(data) - value))
    return idx

def indices_of_the_nearest(data, value):
    distance = np.abs(np.array(data) - value)
    indices = np.where(distance == np.min(distance))[0]
    return indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('yml_fname', help='filename of input parameters (.yml)')
    args = parser.parse_args()
    fname = args.yml_fname
    with open(fname, 'rb') as f:
        param_dict = yaml.full_load(f)

    train_dir = param_dict['train_dir']
    train_datas = param_dict['train_data']
    save_dir_name = param_dict['save_dir_name']
    alpha = param_dict['alpha']
    plot_figure = param_dict['plot_figure']
    random_split = param_dict['random_split']
    lamb = param_dict['lamb']
    target_value = param_dict['target_value']
    threshold = param_dict['threshold']
    std = param_dict['std']
    target_data_csv = param_dict['target_data_csv']
    original_image_num = param_dict['original_image_num']
    increase_rate = param_dict['increase_rate']
    
    save_dir = f'{train_dir}/{save_dir_name}_x{increase_rate}'
    if os.path.exists(save_dir):
        save_dirs = glob(save_dir + '*')
        sorted_save_dirs = natsorted(save_dirs)
        latest_dir = os.path.basename(
            sorted_save_dirs[len(sorted_save_dirs) - 1])
        result = re.sub(r"\D", "", latest_dir)
        if result == '':
            save_dir = os.path.join(train_dir, 'mixup2')
        else:
            save_dir = os.path.join(
                train_dir, f'mixup{int(result) + 1}')
    else:
        if not os.path.exists(os.path.join(train_dir, 'data_org')):
            os.mkdir(os.path.join(train_dir, 'data_org'))
            data_org = glob(os.path.join(train_dir, '*.*'))
            for data in data_org:
                shutil.copy(os.path.join(data),
                            os.path.join(train_dir, 'data_org'))

    os.mkdir(save_dir)
    
    target_keys = ['normalized_mean_slope_gradient', 'altitude_difference', 'mean_slope_gradient_norm_data', 'mean_slope_gradient']
    target_key = target_keys[target_value]

    slope_normalized = False
    if target_key == 'normalized_mean_slope_gradient':
        slope_normalized = True
    
    index_high, index_low, slope_high, slope_low, df, target_df = load_csv_norm(os.path.join(train_dir, train_datas[1].replace('.pkl', '.csv')), target_key, target_data_csv, save_dir, original_image_num)

    imgs_HR, imgs_HR_org = load_imgs(os.path.join(train_dir, train_datas[0]))
    imgs_LR, imgs_LR_org = load_imgs(os.path.join(train_dir, train_datas[1]))

    imgs_HR_mixup, x_HR_trans, x_HR, x_HR_high, x_HR_low, imgs_LR_mixup, x_LR_trans, x_LR, x_LR_high, x_LR_low, mixup_data = mixup_norm(imgs_HR, imgs_HR_org, imgs_LR, imgs_LR_org, index_high, index_low, train_dir, target_df, target_key, slope_normalized)
    
    with open(os.path.join(save_dir, train_datas[0]), 'wb') as f:
        pickle.dump(imgs_HR_mixup, f)
    with open(os.path.join(save_dir, train_datas[1]), 'wb') as f:
        pickle.dump(imgs_LR_mixup, f)

    mixup_data_dir = os.path.join(save_dir, 'mixuped_data')
    os.mkdir(mixup_data_dir)
    
    with open(os.path.join(mixup_data_dir, 'data_HR_mixup.pkl'), 'wb') as f:
        pickle.dump(x_HR_trans, f)
    with open(os.path.join(mixup_data_dir, 'data_LR_mixup.pkl'), 'wb') as f:
        pickle.dump(x_LR_trans, f)
    
    mixup_base_save_dir = os.path.join(save_dir, 'mixup_base_data')
    os.makedirs(mixup_base_save_dir, exist_ok=True)
    
    with open(os.path.join(mixup_base_save_dir, 'data_HR_mixupped.pkl'), 'wb') as f:
        pickle.dump(x_HR, f)
    with open(os.path.join(mixup_base_save_dir, 'data_LR_mixupped.pkl'), 'wb') as f:
        pickle.dump(x_LR, f)
    with open(os.path.join(mixup_base_save_dir, 'data_HR_high.pkl'), 'wb') as f:
        pickle.dump(x_HR_high, f)
    with open(os.path.join(mixup_base_save_dir, 'data_LR_high.pkl'), 'wb') as f:
        pickle.dump(x_LR_high, f)
    with open(os.path.join(mixup_base_save_dir, 'data_HR_low.pkl'), 'wb') as f:
        pickle.dump(x_HR_low, f)
    with open(os.path.join(mixup_base_save_dir, 'data_LR_low.pkl'), 'wb') as f:
        pickle.dump(x_LR_low, f)
    
    subprocess.run(['python', 'bathymetric_map_feature_LR.py', os.path.join(save_dir, 'data_LR.pkl')])

    pick_images(target_df, save_dir, imgs_HR_mixup, imgs_LR_mixup, original_image_num, target_key, os.path.join(train_dir, train_datas[1].replace('.pkl', '.csv')), increase_rate)
    
    if plot_figure:
        data_LR_mixup = pd.read_csv(os.path.join(save_dir, 'data_LR.csv'))
        data_LR_mixup = data_LR_mixup.loc[data_LR_mixup['index'] >= original_image_num * 4]
        os.makedirs(os.path.join(save_dir, 'mixup_results'), exist_ok=True)
        
        count = 0
        for mdata in tqdm(mixup_data):
            vmax = max([np.nanmax(x_LR_high[mdata[1]][mdata[2]]), np.nanmax(x_LR_low[mdata[3]][mdata[4]]), np.nanmax(x_LR[mdata[0]])])
            vmin = min([np.nanmin(x_LR_high[mdata[1]][mdata[2]]), np.nanmin(x_LR_low[mdata[3]][mdata[4]]), np.nanmin(x_LR[mdata[0]])])
            fig = plt.figure(figsize=(20, 5))
            ax1 = fig.add_subplot(1, 3, 1)
            sns.heatmap(x_LR_high[mdata[1]][mdata[2]], cmap='jet', vmax=vmax, vmin=vmin)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title(
                f'slope_grad={df.loc[mdata[6], target_key]}')
            ax2 = fig.add_subplot(1, 3, 2)
            sns.heatmap(x_LR_low[mdata[3]][mdata[4]],
                        cmap='jet', vmax=vmax, vmin=vmin)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title(
                f'slope_grad={df.loc[mdata[7], target_key]}')
            ax3 = fig.add_subplot(1, 3, 3)
            sns.heatmap(x_LR[mdata[0]], cmap='jet', vmax=vmax, vmin=vmin)
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_title(
                f'slope_grad={data_LR_mixup[target_key][mdata[0] + (original_image_num * 4)]}(mixup)')
            plt.savefig(os.path.join(save_dir, f'mixup_results/{count}.jpg'), dpi=100, bbox_inches='tight')
            plt.close()
            count += 1
