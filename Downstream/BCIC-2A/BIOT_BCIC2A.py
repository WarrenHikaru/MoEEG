import math
import random
import os
import torch
from torch import nn
import pytorch_lightning as pl

from functools import partial
import numpy as np
import random
import os 
import tqdm
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

torch.set_float32_matmul_precision('medium')

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

from Modules.Network.utils import Conv1dWithConstraint
from Modules.BIOT.biot import (
    BIOTClassifier,
)

from utils_eval import get_metrics



class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, pretrain_model_choice = 0):
        super().__init__() 
        pretrain_models = ["Modules/BIOT/EEG-PREST-16-channels.ckpt",
                           "Modules/BIOT/EEG-SHHS+PREST-18-channels.ckpt",
                           "Modules/BIOT/EEG-six-datasets-18-channels.ckpt"]
        if pretrain_model_choice == 0: in_channels = 16
        elif pretrain_model_choice == 1: in_channels = 18
        elif pretrain_model_choice == 2: in_channels = 18
        else: raise ValueError("pretrain_model_choice should be 0, 1, or 2")
        
        self.chan_conv      = Conv1dWithConstraint(22, in_channels, 1, max_norm=1)
        model = BIOTClassifier(
                    n_classes=4,
                    # set the n_channels according to the pretrained model if necessary
                    n_channels=in_channels,
                    n_fft=200,
                    hop_length=100,
                )
        model.biot.load_state_dict(torch.load(pretrain_models[pretrain_model_choice]))
        print(f"load pretrain model from {pretrain_models[pretrain_model_choice]}")
        for p in model.biot.transformer.parameters():
            p.requires_grad = False
        self.feature        = model
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity=True
        
    
    def forward(self, x):
        B, C, T = x.shape
        if T%200!=0: 
            x = x[:,:,0:T-T%200]
            T = T-T%200
        x = x.float()
        x = self.chan_conv(x)
        pred = self.feature(x)
        return x, pred

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)
        
        return loss
        
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
            
        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)
        
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, False)
        
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)
        
        self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters())+
            list(self.feature.parameters()),
            weight_decay=0.01)#
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )


class EEGDataset(Dataset):
    def __init__(self, root_dir, subject_ids):
        self.data = []
        self.labels = []

        for sub_id in subject_ids:
            sub_dir = os.path.join(root_dir, f"sub{sub_id}")
            data = torch.load(os.path.join(sub_dir, "data.pt"))  # 形状: [样本数, 通道数, 时间点]
            labels = torch.load(os.path.join(sub_dir, "label.pt"))  # 形状: [样本数]
            self.data.append(data)
            self.labels.append(labels)
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    batch_size = 16
    max_epochs = 100
    max_lr = 8e-4
    data_root = "BCIC_2aT_0_38HZ"
    all_subjects = [1,2,3,4,5,6,7,8,9]
    tested_subject = []

    for seed in [7,718,1029]:

        seed_torch(seed)

        for test_sub in all_subjects:
            if test_sub in tested_subject:
                continue
            train_subjects = [sub for sub in all_subjects if sub != test_sub]
            print(f"\nTest: sub{test_sub}, Train: sub {train_subjects}")

            train_dataset = EEGDataset(data_root, train_subjects)
            test_dataset = EEGDataset(data_root, [test_sub])

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,

            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                persistent_workers=True
            )

            steps_per_epoch = math.ceil(len(train_loader.dataset) / batch_size)

            model = LitEEGPTCausal()
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
            logger = [
                pl_loggers.TensorBoardLogger(f'./logs/BIOT/Seed{seed}', name="LOSO_BCIC2a_tb", version=f"test_sub{test_sub}"),
                pl_loggers.CSVLogger(f'./logs/BIOT/Seed{seed}', name="LOSO_BCIC2a_csv", version=f"test_sub{test_sub}")
        ]

            trainer = pl.Trainer(
                accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                max_epochs=max_epochs,
                callbacks=[lr_monitor],
                logger=logger,
                enable_checkpointing=False
            )

            trainer.fit(model, train_loader, test_loader)