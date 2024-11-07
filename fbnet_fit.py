import json
from pathlib import Path

from torch.optim.lr_scheduler import MultiStepLR

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from nni.nas.space import model_context
from nni.nas.evaluator.pytorch import DataLoader

from datasets import prepare_cifar_dataset, prepare_imagenet_dataset
from fbnet.lightning import FBNetClassification
from fbnet.model import FBNet
from fbnet.optimizer import SGDMomentum


def fbnet_fit():
    # Prepare datasets
    # train_data, valid_data, num_classes = prepare_cifar_dataset()
    train_data, valid_data, num_classes = prepare_imagenet_dataset(nni_trace=True)

    # Load searched architecture
    # search_experiment_dir = "experiments"
    # search_experiment_name = "search-imagenet"
    # search_experiment_version = "batch=480"

    # searched_arch_filename = "searched-arch.json"
    # searched_arch_dirpath = f"{search_experiment_dir}/{search_experiment_name}/{search_experiment_version}"
    # searched_arch_file_path = f"{searched_arch_dirpath}/{searched_arch_filename}"

    searched_arch_file_path = "fbnet/fbnet-a.json"
    with open(searched_arch_file_path, "r") as f:
        exported_arch = json.load(f)

    # Get model from searched architecture
    with model_context(exported_arch):
        final_model = FBNet(num_classes)

    # Define evaluator
    fast_dev_run = False

    num_gpus = 7
    batch_size_per_gpu = 64  # This should fit in RTX 3090 (~20GB, this varies as per the searched arch)
    batch_size = batch_size_per_gpu * num_gpus  # 256 = 64 * 4

    # max_epochs = 360  # FBNet default
    # lr_scheduler_milestones = [90, 180, 270]  # FBNet default
    # Change this for faster training
    max_epochs = 360
    lr_scheduler_milestones = [int(max_epochs * 1/4), int(max_epochs * 2/4), int(max_epochs * 3/4)]

    # max_epochs = 150
    # lr_scheduler_milestones = [60, 90, 120]

    # FBNet paper uses batch_size=256, learning_rate=1e-1, weight_decay=4e-5
    # Set hyperparameters by linear scaling rule
    scale = batch_size / 256  # Using linear scaling rule
    learning_rate = 1e-1 * scale  # Using linear scaling rule
    weight_decay = 4e-5

    # ImageNet gradual warmup learning rate convention:
    #   Warm up 5 epochs per 120 epochs training
    # gradual_warmup_epochs = 5 * (max_epochs / 120)
    gradual_warmup_epochs = 5

    # Set TensorBoard log and model checkpoint directory
    experiments_dir = "experiments"
    experiment_name = "fbnet-v1-fit"
    experiment_version = f"fbnet-a_batch={batch_size}_epoch={max_epochs}"

    # TODO(donghyeon): handle added version suffix cases (ex: "last_v1.ckpt")
    # Set resume=True to resume stopped experiments
    auto_resume = True
    resume_ckpt_filename = "last.ckpt" # (or "epoch=5-step=10014.ckpt")
    checkpoint_dirpath = f"{experiments_dir}/{experiment_name}/{experiment_version}"
    checkpoint_file_path = f"{checkpoint_dirpath}/{resume_ckpt_filename}"
    # Expected checkpoint file path: ./experiments/fit_imagenet/batch=1024/last.ckpt
    resume = Path(checkpoint_file_path).exists() if auto_resume else False

    train_loader = DataLoader(train_data, batch_size=batch_size_per_gpu, num_workers=8,
                              pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size_per_gpu, num_workers=8,
                              pin_memory=True, persistent_workers=True)

    evaluator = FBNetClassification(
        num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer=SGDMomentum,
        gradual_warmup_epochs=gradual_warmup_epochs,
        lr_scheduler_fn=lambda optimizer: MultiStepLR(
            optimizer=optimizer,
            milestones=lr_scheduler_milestones,
            gamma=0.1,
        ),
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        trainer_kwargs=dict(
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=num_gpus,
            strategy=DDPStrategy(find_unused_parameters=False),
            sync_batchnorm=True,
            precision="16-mixed",
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
                ModelCheckpoint(dirpath=checkpoint_dirpath, save_last=True),
            ],
            logger=TensorBoardLogger(experiments_dir, name=experiment_name, version=experiment_version),
            fast_dev_run=fast_dev_run,
        ),
        fit_kwargs=dict(ckpt_path=checkpoint_file_path) if resume else None,
    )

    # Final training
    evaluator.fit(final_model)


if __name__ == "__main__":
    fbnet_fit()
