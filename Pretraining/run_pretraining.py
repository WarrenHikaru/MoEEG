# Training in 256Hz data and 4s
import torch
from pytorch_lightning import loggers as pl_loggers

from engine_pretraining import *
from configs import *
from pytorch_lightning.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision("medium")

seed_torch(815)


# init model

model = LitEEGPT(get_config(**(MODELS_CONFIGS[tag])))

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

checkpoint_callback = ModelCheckpoint(
    dirpath='./Result/Model/',
    filename=f"MoEEG_{tag}",
    save_weights_only=False,
    save_last=True
)

callbacks = [lr_monitor, checkpoint_callback]

trainer = pl.Trainer(strategy='auto', devices=devices, max_epochs=max_epochs, callbacks=callbacks,
                    precision="16-mixed",
                     logger=[pl_loggers.TensorBoardLogger('./Result/logs/', name=f"MoEEG_{tag}_tb"),
                             pl_loggers.CSVLogger('./Result/logs/', name=f"MoEEG_{tag}_csv")])

if __name__ == "__main__":
    trainer.fit(model, train_loader, valid_loader)