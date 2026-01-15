import os
import sys
import warnings
import mne
import numpy as np
import scipy.io
from braindecode.preprocessing import exponential_moving_standardize

# 路径设置（与你的目录结构一致）
current_path = os.path.abspath(os.path.dirname(__file__))
root_path = "../../dataset/downstream"
sys.path.append(current_path)


def EMS(data):
    """保留原始EMS标准化函数"""
    new_x = []
    for x in data:
        new_x.append(exponential_moving_standardize(x))
    return np.array(new_x)


def load_bci2a_train(gdf_path, tmin=0, tmax=4, bandpass=[8, 30], resample=128):
    """加载训练集（A0*T.gdf）：从.gdf提取数据和标签"""
    # 读取.gdf
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Channel names are not unique")
        raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)

    # 提取事件（训练集标签在.gdf中）
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    # 筛选769-772的任务事件
    target_strs = ['769', '770', '771', '772']
    target_codes = [event_id[s] for s in target_strs if s in event_id]
    valid_mask = np.isin(events[:, 2], target_codes)
    valid_events = events[valid_mask]

    # 预处理
    if bandpass:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            raw.filter(bandpass[0], bandpass[1], verbose=False)
    if resample and resample != raw.info['sfreq']:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            raw.resample(resample, verbose=False)

    # 提取epochs和标签
    epochs = mne.Epochs(raw, valid_events, tmin=tmin, tmax=tmax, baseline=None, verbose=False, preload=True)
    x_data = epochs.get_data()
    # 标签映射为0-3
    str2label = {'769': 0, '770': 1, '771': 2, '772': 3}
    code2str = {v: k for k, v in event_id.items()}
    y_labels = np.array([str2label[code2str[code]] for code in valid_events[:, 2]])

    return x_data, y_labels


def load_bci2a_test(gdf_path, mat_label_path, tmin=0, tmax=4, bandpass=[8, 30], resample=128):
    """加载测试集（A0*E.gdf）：.gdf提数据，.mat提标签（标准BCIC 2a格式）"""
    # 1. 从.gdf提取数据（无标签，仅信号）
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Channel names are not unique")
        raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)

    # 提取事件（仅需事件位置，标签从.mat来）
    events, _ = mne.events_from_annotations(raw, verbose=False)
    # 筛选测试集任务事件（标准BCIC 2a测试集事件编码为783，对应任务开始）
    valid_mask = events[:, 2] == [v for v in events[:, 2] if v not in [1023, 1072, 276, 277, 32766, 768]][0]
    valid_events = events[valid_mask]

    # 预处理（与训练集一致）
    if bandpass:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            raw.filter(bandpass[0], bandpass[1], verbose=False)
    if resample and resample != raw.info['sfreq']:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            raw.resample(resample, verbose=False)

    # 提取epochs数据
    epochs = mne.Epochs(raw, valid_events, tmin=tmin, tmax=tmax, baseline=None, verbose=False, preload=True)
    x_data = epochs.get_data()

    # 2. 从官网下载的.mat标签文件提取标签
    mat_data = scipy.io.loadmat(mat_label_path)
    # 标准BCIC 2a的.mat标签存在'classlabel'或'y'字段中
    if 'classlabel' in mat_data:
        y_labels = mat_data['classlabel'].reshape(-1) - 1  # 官网标签1-4→转为0-3
    elif 'y' in mat_data:
        y_labels = mat_data['y'].reshape(-1) - 1
    else:
        raise KeyError(".mat标签文件中未找到'classlabel'或'y'字段，请确认是标准BCIC 2a标签文件")

    # 确保数据和标签数量匹配（标准测试集144个样本）
    assert len(x_data) == len(y_labels), f"数据样本数{len(x_data)}与标签数{len(y_labels)}不匹配"

    return x_data, y_labels


def process_all_bci2a(subjects=range(1, 10), tmin=0, tmax=4, bandpass=[8, 30], resample=128):
    """处理所有被试：训练集（gdf）+测试集（gdf+mat标签）"""
    # 数据路径（请确保.mat标签文件放在与.gdf同目录）
    raw_dir = os.path.join(root_path, 'Raw_data', 'BCICIV_2a_gdf')
    # 保存路径
    save_dir = os.path.join(root_path, 'Data', f'BCIC_2a_{bandpass[0]}_{bandpass[1]}HZ' if bandpass else 'BCIC_2a')
    os.makedirs(save_dir, exist_ok=True)

    for sub in subjects:
        print(f"\n=== 处理被试{sub} ===")
        # ---------------------- 处理训练集（A0{sub}T.gdf）----------------------
        train_gdf = os.path.join(raw_dir, f'A0{sub}T.gdf')
        if not os.path.exists(train_gdf):
            print(f"警告：训练集文件 {train_gdf} 不存在，跳过")
            continue

        try:
            train_x, train_y = load_bci2a_train(train_gdf, tmin, tmax, bandpass, resample)
            train_x = EMS(train_x)
            print(f"训练集：x_shape={train_x.shape}, y_shape={train_y.shape}")
        except Exception as e:
            print(f"训练集处理失败：{str(e)}")
            continue

        # ---------------------- 处理测试集（A0{sub}E.gdf + A0{sub}E.mat）----------------------
        test_gdf = os.path.join(raw_dir, f'A0{sub}E.gdf')
        test_mat = os.path.join(raw_dir, f'A0{sub}E.mat')  # 官网下载的标签文件
        if not os.path.exists(test_gdf) or not os.path.exists(test_mat):
            print(f"警告：测试集文件（{test_gdf} 或 {test_mat}）不存在，跳过")
            continue

        try:
            test_x, test_y = load_bci2a_test(test_gdf, test_mat, tmin, tmax, bandpass, resample)
            test_x = EMS(test_x)
            print(f"测试集：x_shape={test_x.shape}, y_shape={test_y.shape}")
        except Exception as e:
            print(f"测试集处理失败：{str(e)}")
            continue

        # ---------------------- 保存数据（与原始代码格式一致）----------------------
        # 训练集保存
        train_save_dir = os.path.join(save_dir, f'sub{sub}_train')
        os.makedirs(train_save_dir, exist_ok=True)
        scipy.io.savemat(os.path.join(train_save_dir, 'Data.mat'), {'x_data': train_x, 'y_data': train_y})

        # 测试集保存
        test_save_dir = os.path.join(save_dir, f'sub{sub}_test')
        os.makedirs(test_save_dir, exist_ok=True)
        scipy.io.savemat(os.path.join(test_save_dir, 'Data.mat'), {'x_data': test_x, 'y_data': test_y})

        print(f"被试{sub} 处理完成！")

    print("\n所有被试处理结束！")


if __name__ == '__main__':
    # 1. 请先从BCIC官网下载测试集标签文件（A01E.mat~A09E.mat），放在与.gdf同目录
    # 官网地址：https://bnci-horizon-2020.eu/database/data-sets
    # 2. 运行处理
    process_all_bci2a(
        subjects=range(1, 10),
        tmin=0,
        tmax=4,
        bandpass=[8, 30],  # 运动想象最优频段
        resample=128  # 常用重采样率
    )