import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import argparse

def get_slope(h, delta=200):
    nx = 16
    ny = 16
    dx = delta
    dy = delta

    s = np.zeros((nx, ny))

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            sx = (h[i-1, j+1]+h[i-1, j]+h[i-1, j-1])-(h[i+1,j+1]+h[i+1, j]+h[i+1, j-1])
            sx = sx/(6.0*dx)
            sy = (h[i-1, j-1]+h[i, j-1]+h[i+1, j-1])-(h[i-1, j+1]+h[i,j+1]+h[i+1, j+1])
            #xy = sy/(6.0*dy)
            sy = sy/(6.0*dy)
            s[i,j] = np.sqrt(sx*sx+sy*sy)
    return np.mean(s)

def get_depth_range(h):
    return h.max() - h.min()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file',help='filename of input pickle data')
    args = parser.parse_args()

    in_file = Path(args.in_file)
    with open(in_file, 'rb') as f:
        data = pickle.load(f)

    norm_range = [np.nanmin(data), np.nanmax(data)]
    data_norm = (data - norm_range[0]) / (norm_range[1] - norm_range[0])

    res = []
    for i in tqdm.tqdm(range(data.shape[0])):
        h = data[i]
        slope = get_slope(h)
        slope_norm = get_slope(data_norm[i], delta = 200.0/abs(np.diff(norm_range)[0]))
        drange = get_depth_range(h)
        res.append({'index':i, 
                    'mean_slope_gradient':slope, 
                    'altitude_difference':drange, 
                    'mean_altitude':h.mean(),
                    'mean_slope_gradient_norm_data': slope_norm
                    })

    res = pd.DataFrame(res)
    slmax = res['mean_slope_gradient'].max()
    slmin = res['mean_slope_gradient'].min()
    res['normalized_mean_slope_gradient'] = (res['mean_slope_gradient'] - slmin) / (slmax - slmin)

    res.to_csv(in_file.with_suffix('.csv'))

    plt.scatter(x='altitude_difference', y='mean_altitude',c='mean_slope_gradient', data=res)
    plt.colorbar(label='mean_slope_gradient')
    plt.xlabel('altitude_difference')
    plt.ylabel('mean_altitude')
    # plt.legend()
    # plt.show()
    plt.savefig(in_file.parent / 'scatter1.png')

    plt.close()
    plt.cla()
    plt.clf()

    plt.scatter(x='altitude_difference', y='mean_slope_gradient', data=res)
    plt.xlabel('altitude_difference')
    plt.ylabel('mean_slope_gradient')
    # plt.show()
    plt.savefig(in_file.parent / 'scatter2.png')

    plt.close()
    plt.cla()
    plt.clf()
    df_oki = pd.read_csv('../../../oki_lr_inter/oki_lr_inter/test/data_LR.csv')
    max_num = max(df_oki['mean_slope_gradient'])
    bins_num = 0
    bins = [0]
    while(True):
        bins_num += 0.02
        bins.append(bins_num)
        if bins_num > max_num:
            break
    res['mean_slope_gradient'].hist(bins=bins, alpha=0.7)
    df_oki['mean_slope_gradient'].hist(bins=bins, alpha=0.7)
    plt.savefig(in_file.parent / 'slope_hist.png')

    plt.close()
    plt.cla()
    plt.clf()

    res['normalized_mean_slope_gradient'].hist(bins=20)
    plt.savefig(in_file.parent / 'slope_norm_hist.png')

    plt.close()
    plt.cla()
    plt.clf()

    res['altitude_difference'].hist(bins=20)
    plt.savefig(in_file.parent / 'alt_dif_hist.png')

    plt.close()
    plt.cla()
    plt.clf()

    res['mean_altitude'].hist(bins=20)
    plt.savefig(in_file.parent / 'mean_alst_hist.png')

    plt.close()
    plt.cla()
    plt.clf()
