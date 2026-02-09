import torch
import torchvision
import math
import random


def load_fn(x):
    x = torch.load(x)

    window_length = 4 * 256
    data_length = x.shape[1]

    # Calculate the maximum starting index for the windows
    max_start_index = data_length - window_length

    # Generate random indices
    if max_start_index > 0:
        index = random.randint(0, max_start_index)
        x = x[:, index:index + window_length]
    x = x.to(torch.float)
    return x


max_epochs = 30
max_lr = 6e-5
batch_size = 16
devices = [0]

train_dataset = torchvision.datasets.DatasetFolder(
    root="..\\Datasets\\datasets_pretraining\\merged\\TrainFolder\\",
    loader=load_fn, extensions=['.edf'])
valid_dataset = torchvision.datasets.DatasetFolder(
    root="..\\Datasets\\datasets_pretraining\\merged\\ValidFolder\\",
    loader=load_fn, extensions=['.edf'])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True,persistent_workers=True,drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=2, shuffle=False,persistent_workers=True,drop_last=True)

steps_per_epoch = math.ceil(len(train_loader) / len(devices))

tag = "base"
MODELS_CONFIGS = {
    "base": {
        "embed_dim": 64, "embed_num": 4, "depth": [8, 8, 8], "num_heads": 4},
    "large": {
        "embed_dim": 512, "embed_num": 4, "depth": [8, 8, 8], "num_heads": 8},

}


def get_config(embed_dim=512, embed_num=4, depth=[8, 8, 8], num_heads=4):
    models_configs = {
        'encoder': {
            'embed_dim': embed_dim,
            'embed_num': embed_num,
            'depth': depth[0],
            'num_heads': num_heads,
        },
        'predictor': {
            'embed_dim': embed_dim,
            'embed_num': embed_num,
            'predictor_embed_dim': embed_dim,
            'depth': depth[1],
            'num_heads': num_heads,
        },
        'reconstructor': {
            'embed_dim': embed_dim,
            'embed_num': embed_num,
            'reconstructor_embed_dim': embed_dim,
            'depth': depth[2],
            'num_heads': num_heads,
        },
    }
    return models_configs



