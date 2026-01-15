import os
import sys
import mne
import torch
import numpy as np
from braindecode.preprocessing import exponential_moving_standardize

current_path = os.path.abspath(os.path.dirname(__file__))
root_path = "Raw_data"
sys.path.append(current_path)


def EMS(data):
    new_x = []
    for x in data:
        new_x.append(exponential_moving_standardize(x))
    return np.array(new_x)


def Load_BCIC_2b_raw_data(tmin=0, tmax=4, bandpass=[0, 38]):
    data_path = os.path.join(root_path,'BCICIV_2b_gdf')
    if bandpass is None:
        main_save_path = os.path.join(root_path, r'Data', 'BCIC_2b')
    else:
        main_save_path = os.path.join(root_path, r'Data', f'BCIC_2b_{bandpass[0]}_{bandpass[1]}HZ')

    if not os.path.exists(main_save_path):
        os.makedirs(main_save_path)

    for sub in range(1, 10):
        print(f"\nProcess subject{sub}...")

        sub_folder = os.path.join(main_save_path, f'sub{sub}')
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)

        load_raw_data = LoadBCIC_2b(data_path, sub, tmin, tmax, bandpass)
        train_x, train_y = load_raw_data.get_train_data()
        train_x = EMS(train_x)

        print(f"Subject{sub} data shape:")
        print(f"  Data: {train_x.shape}")
        print(f"  Label: {train_y.shape}")

        data_save_path = os.path.join(sub_folder, 'data.pt')
        label_save_path = os.path.join(sub_folder, 'label.pt')

        torch.save(torch.from_numpy(train_x), data_save_path)
        torch.save(torch.from_numpy(train_y), label_save_path)


class LoadBCIC_2b:

    def __init__(self, path, subject, tmin=0, tmax=4, bandpass=None):
        self.tmin = tmin
        self.tmax = tmax
        self.bandpass = bandpass
        self.subject = subject
        self.path = path
        self.train_suffix = ['1', '2', '3']
        self.target_stim_codes = {'769': 0, '770': 1}
        self.channels_to_remove = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        self.sfreq = 250
        self.time_points = int((tmax - tmin) * self.sfreq) + 1

    def get_train_data(self):
        data = []
        label = []
        for se in self.train_suffix:
            data_name = f'B0{self.subject}0{se}T.gdf'
            data_path = os.path.join(self.path, data_name)
            data_x, data_y = self.get_epoch_and_label(data_path)

            data.extend(data_x)
            label.extend(data_y)

        return np.array(data), np.array(label).reshape(-1)

    def get_epoch_and_label(self, data_path):

        raw_data = mne.io.read_raw_gdf(data_path, preload=True)
        events, events_id = mne.events_from_annotations(raw_data)
        target_event_ids = {k: v for k, v in events_id.items() if k in self.target_stim_codes.keys()}

        epochs = mne.Epochs(
            raw_data,
            events,
            event_id=target_event_ids,
            tmin=self.tmin,
            tmax=self.tmax,
            event_repeated='drop',
            baseline=None,
            preload=True,
            proj=False,
            reject_by_annotation=False
        )

        if self.bandpass is not None:
            epochs.filter(self.bandpass[0], self.bandpass[1], method='iir')


        epochs = epochs.drop_channels(self.channels_to_remove)
        eeg_data = epochs.get_data() * 1e6

        labels = []
        for event_id in epochs.events[:, 2]:
            stim_code = next(k for k, v in events_id.items() if v == event_id)
            labels.append(self.target_stim_codes[stim_code])
        labels = np.array(labels)

        assert len(eeg_data) == len(labels)
        return eeg_data, labels


if __name__ == '__main__':
    Load_BCIC_2b_raw_data()