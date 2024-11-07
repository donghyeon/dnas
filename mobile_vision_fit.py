from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import prepare_cifar_dataset, prepare_imagenet_dataset
from fbnet.mobile_vision import MobileVisionFBNet, MobileVisionLightningModule, RASampler

class RASamplerLightningDataModule(LightningDataModule):
    def __init__(self, train_dataset: DataLoader, valid_dataset: DataLoader,
                 batch_size_per_gpu: int, num_workers: int = 8,
                 ra_sampler: bool = False, ra_repetitions: int = 4):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.batch_size_per_gpu = batch_size_per_gpu
        self.num_workers = num_workers
        self.ra_sampler = ra_sampler
        self.ra_repititions = ra_repetitions
    
    def train_dataloader(self):
        if self.ra_sampler:
            train_sampler = RASampler(self.train_dataset, repetitions=self.ra_repititions)
        else:
            train_sampler = DistributedSampler(self.train_dataset)
        return DataLoader(self.train_dataset, batch_size=self.batch_size_per_gpu, num_workers=self.num_workers,
                          sampler=train_sampler,
                          pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size_per_gpu, num_workers=self.num_workers,
                          sampler=DistributedSampler(self.valid_dataset),
                          pin_memory=True)


def mobile_cv_fit():
    # Prepare datasets
    # train_data, valid_data, num_classes = prepare_cifar_dataset()
    train_data, valid_data, num_classes = prepare_imagenet_dataset()

    # Define evaluator
    fast_dev_run = False

    num_gpus = 7
    batch_size_per_gpu = 384
    batch_size = batch_size_per_gpu * num_gpus  # 256 = 64 * 4

    # FBNet paper uses batch_size=256, learning_rate=1e-1, weight_decay=4e-5
    # But we change the defaults according to the torchvision training recipe
    #   recipe: batch_size=1024, learning_rate=0.5, weight_decay=2e-5, norm_weight_decay=0.0
    # Set hyperparameters by linear scaling rule
    scale = batch_size / 1024  # Using linear scaling rule
    learning_rate = 5e-1 * scale  # Using linear scaling rule
    weight_decay = 2e-5
    norm_weight_decay = 0.0
    
    # max_epochs = 360  # FBNet default
    # lr_scheduler_milestones = [90, 180, 270]  # FBNet default
    # Change this according to the torchvision training recipe
    max_epochs = 360

    # ImageNet gradual warmup learning rate convention for stable training
    gradual_warmup_epochs = 5

    # Other traning recipes
    auto_augment = "ta_wide"  # located at datasets.py
    random_erase = 0.1  # located at datasets.py

    label_smoothing = 0.1

    mixup_alpha = 0.2
    cutmix_alpha = 1.0

    ra_sampler = False
    ra_reps = 4

    model_ema = True
    model_ema_steps = 32
    model_ema_decay = 0.99998

    # Set TensorBoard log and model checkpoint directory
    experiments_dir = "experiments"
    experiment_name = "fbnet-v1-recipe-fit"
    experiment_version = f"mobile-vision_fbnet-a_batch={batch_size}_epoch={max_epochs}"

    if model_ema:
        experiment_version += "_model-ema=true"
    if ra_sampler:
        experiment_version += "_ra-sampler=true"

    # TODO(donghyeon): handle added version suffix cases (ex: "last_v1.ckpt")
    # Set resume=True to resume stopped experiments
    auto_resume = True
    resume_ckpt_filename = "last.ckpt" # (or "epoch=5-step=10014.ckpt")
    checkpoint_dirpath = f"{experiments_dir}/{experiment_name}/{experiment_version}"
    checkpoint_file_path = f"{checkpoint_dirpath}/{resume_ckpt_filename}"
    # Expected checkpoint file path: ./experiments/fit_imagenet/batch=1024/last.ckpt
    resume = Path(checkpoint_file_path).exists() if auto_resume else False

    # train_loader = DataLoader(train_data, batch_size=batch_size_per_gpu, num_workers=10,
    #                           pin_memory=True, persistent_workers=True)
    # valid_loader = DataLoader(valid_data, batch_size=batch_size_per_gpu, num_workers=10,
    #                           pin_memory=True)

    datamodule = RASamplerLightningDataModule(train_dataset=train_data, valid_dataset=valid_data,
                                              batch_size_per_gpu=batch_size_per_gpu,
                                              num_workers=12,
                                              ra_sampler=ra_sampler, ra_repetitions=ra_reps)
    
    model = MobileVisionFBNet("fbnet_a")
    lightning_module = MobileVisionLightningModule(
        model=model,
        num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        norm_weight_decay=norm_weight_decay,
        max_epochs=max_epochs,
        gradual_warmup_epochs=gradual_warmup_epochs,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        label_smoothing=label_smoothing,
        model_ema=model_ema,
        model_ema_steps=model_ema_steps,
        model_ema_decay=model_ema_decay,
        batch_size=batch_size,
    )
    
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=num_gpus,
        strategy="ddp_find_unused_parameters_true",
        use_distributed_sampler=False,
        sync_batchnorm=True,
        precision="16-mixed",
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(dirpath=checkpoint_dirpath, save_last=True),
        ],
        logger=TensorBoardLogger(experiments_dir, name=experiment_name, version=experiment_version),
        fast_dev_run=fast_dev_run,
    )
    
    torch.set_float32_matmul_precision("medium")
    trainer.fit(
        model=lightning_module,
        datamodule=datamodule,
        ckpt_path=checkpoint_file_path if resume else None,
    )


if __name__ == "__main__":
    mobile_cv_fit()
