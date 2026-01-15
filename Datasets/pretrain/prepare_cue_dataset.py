"""
[B,60,T] -> [58,1024]
Cue Data Process
"""

import numpy as np
import torch
import os
import random
from glob import glob

use_channels_names = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2'
]

original_60_channels = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2'
]

assert len(original_60_channels) == 60
use_channel_indices = [original_60_channels.index(name) for name in use_channels_names]
assert len(use_channel_indices) == 58

def temporal_interpolation(x, desired_length):
    x = x - x.mean(dim=-1, keepdim=True)  # 时间维度归一化
    return torch.nn.functional.interpolate(x, size=desired_length, mode='nearest')

def process_eeg_data(raw_data, target_time=256, concat_size=4):
    if isinstance(raw_data, np.ndarray):
        raw_data = torch.from_numpy(raw_data).float()

    filtered = raw_data[:, use_channel_indices, :]
    resampled = temporal_interpolation(filtered, desired_length=target_time)

    total = resampled.shape[0]
    valid = (total // concat_size) * concat_size
    if valid < total:
        resampled = resampled[:valid]


    num_groups = valid // concat_size
    concatenated_list = []
    for i in range(num_groups):
        group = resampled[i * concat_size: (i + 1) * concat_size, :, :]  # 形状[4, 58, 256]
        concatenated_group = torch.cat([group[j] for j in range(concat_size)], dim=-1)  # [58, 1024]
        concatenated_list.append(concatenated_group)

    concatenated = torch.stack(concatenated_list, dim=0)
    return concatenated


def process_all_files(folder_a, train_folder, test_folder, train_ratio=0.9,
                      target_time=256, concat_size=4, random_seed=42):

    random.seed(random_seed)
    global_sample_count = 1

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    data_files = glob(os.path.join(folder_a, "*data.npy"))
    random.shuffle(data_files)
    split_idx = int(len(data_files) * train_ratio)
    train_files = data_files[:split_idx]
    test_files = data_files[split_idx:]
    global_sample_count = process_and_split_samples(
        file_list=train_files,
        save_folder=train_folder,
        target_time=target_time,
        concat_size=concat_size,
        start_count=global_sample_count
    )

    process_and_split_samples(
        file_list=test_files,
        save_folder=test_folder,
        target_time=target_time,
        concat_size=concat_size,
        start_count=global_sample_count
    )


def process_and_split_samples(file_list, save_folder, target_time, concat_size, start_count):
    current_count = start_count
    for file_path in file_list:
        raw_data = np.load(file_path)
        processed_data = process_eeg_data(raw_data, target_time, concat_size)
        sample_num = processed_data.shape[0]

        for i in range(sample_num):
            single_sample = processed_data[i]
            single_sample_fp16 = single_sample.to(torch.float16)

            save_name = f"Cue_Smoking_{current_count}_data.edf"
            save_path = os.path.join(save_folder, save_name)

            torch.save(single_sample_fp16, save_path)
            current_count += 1

    return current_count


if __name__ == "__main__":

    folder_a = "..\\Datasets\\EEG\\Cue-reactivity\\smoking\\prepare_data"
    train_folder = "../../datasets_pretraining/Cue/Smoking/TrainFolder"
    valid_folder = "../../datasets_pretraining/Cue/Smoking/ValidFolder"

    process_all_files(
        folder_a=folder_a,
        train_folder=train_folder,
        test_folder=valid_folder,
        train_ratio=0.9
    )