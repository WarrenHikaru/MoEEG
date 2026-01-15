import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
# root_path = os.path.split(current_path)[0]
root_path = "Raw_data"

sys.path.append(current_path)

import LoadData
import numpy as np
import scipy.linalg
import scipy.io
import scipy.sparse
import scipy.signal as signal
from braindecode.preprocessing import exponential_moving_standardize
# from einops import rearrange
from sklearn.model_selection import train_test_split
import torch

def EMS(data):
    new_x = []

    for x in data:
        new_x.append(exponential_moving_standardize(x))

    return np.array(new_x)

def Load_BCIC_2a_T_data(tmin=0, tmax=4, bandpass=[0, 38], resample=None):
    '''
    Load BCIC 2a data end with T's .gdf file
    Args:
        tmin: Time start
        tmax: Time end
        bandpass: band-pass filtering, range [low, high]
        resample: resample data  frequency
    '''
    data_path = os.path.join(root_path, 'BCICIV_2a_gdf')

    if bandpass is None:
        SAVE_path = os.path.join(root_path, 'Data', 'BCIC_2aT')
    else:
        SAVE_path = os.path.join(root_path, 'Data', f'BCIC_2aT_{bandpass[0]}_{bandpass[1]}HZ')

    if not os.path.exists(SAVE_path):
        os.makedirs(SAVE_path)

    for sub in range(1, 10):
        data_name = f'A0{sub}T.gdf'
        data_loader = LoadData.LoadBCIC(data_name, data_path)
        data = data_loader.get_epochs(tmin=tmin, tmax=tmax, bandpass=bandpass, resample=resample)

        train_x = np.array(data['x_data'])
        train_y = np.array(data['y_labels']).reshape(-1)

        train_x = EMS(train_x)

        print(f'Sub{sub} data shape:')
        print(f'train_x: {train_x.shape}')
        print(f'train_y: {train_y.shape}')

        sub_save_path = os.path.join(SAVE_path, f'sub{sub}')
        if not os.path.exists(sub_save_path):
            os.makedirs(sub_save_path)

        torch.save(torch.from_numpy(train_x), os.path.join(sub_save_path, "data.pt"))
        torch.save(torch.from_numpy(train_y), os.path.join(sub_save_path, "label.pt"))

        print(f'Sub{sub} data and label has been saved as torch file')



if __name__ == '__main__':
    Load_BCIC_2a_T_data()