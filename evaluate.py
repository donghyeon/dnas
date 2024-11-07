import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from nni.nas.space import model_context

from datasets import prepare_imagenet_dataset, prepare_imagenet_tta_valid_dataset
from fbnet.mobile_vision import MobileVisionFBNet, MobileVisionLightningModule
from fbnet.model import FBNet


def mobile_cv_evaluate():
    # Prepare datasets
    train_data, valid_data, num_classes = prepare_imagenet_dataset()
    tta_valid_data = prepare_imagenet_tta_valid_dataset()

    num_gpus = 7
    batch_size_per_gpu = 64

    # Set model checkpoint directory
    experiments_dir = "experiments"
    experiment_name = "fbnet-v1-sync-bn-fit"
    experiment_version = f"mobile-vision_fbnet-a_batch=448_epoch=120_lr-scheduler=cosine"
    resume_ckpt_filename = "last.ckpt" # (or "epoch=5-step=10014.ckpt")

    checkpoint_dirpath = f"{experiments_dir}/{experiment_name}/{experiment_version}"
    checkpoint_file_path = f"{checkpoint_dirpath}/{resume_ckpt_filename}"
    # Expected checkpoint file path example: ./experiments/fit_imagenet/batch=1024/last.ckpt

    valid_loader = DataLoader(valid_data, batch_size=batch_size_per_gpu, num_workers=8,
                              pin_memory=True)
    tta_valid_loader = DataLoader(tta_valid_data, batch_size=batch_size_per_gpu, num_workers=8,
                                  pin_memory=True)
    
    # 1. NNI implementation
    # searched_arch_file_path = "fbnet/fbnet-a.json"
    # with open(searched_arch_file_path, "r") as f:
    #     exported_arch = json.load(f)

    # # Get model from searched architecture
    # with model_context(exported_arch):
    #     model = FBNet(num_classes)

    # 2. MobileVision implementation
    # Change this to True when you want to verify the performance of the paper
    # Authors doesn't give exact training code, so we couldn't reproduce the same result.
    from_public_pretrained=False

    model = MobileVisionFBNet("fbnet_a", from_public_pretrained=from_public_pretrained)    
    
    # NNI models can also use the MobileVisionLightningModule when evaluating
    lightning_module = MobileVisionLightningModule(
        model=model,
        num_classes=num_classes,
    )
    
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(
        accelerator="gpu",
        devices=num_gpus,
        strategy="ddp",
        precision="16-mixed",
    )

    # For accurate measurement, use devices=1 and num_nodes=1
    # trainer = Trainer(
    #     accelerator="gpu",
    #     devices=1,
    #     num_nodes=1,
    #     precision="16-mixed",
    # )

    if checkpoint_file_path and not Path(checkpoint_file_path).exists():
        raise(f"Checkpoint file not found: {checkpoint_file_path}")

    trainer.validate(
        model=lightning_module,
        dataloaders=valid_loader,
        ckpt_path=checkpoint_file_path if not from_public_pretrained else None,
    )
    
    trainer.test(
        model=lightning_module,
        dataloaders=tta_valid_loader,
        ckpt_path=checkpoint_file_path if not from_public_pretrained else None,
    )


if __name__ == "__main__":
    mobile_cv_evaluate()
