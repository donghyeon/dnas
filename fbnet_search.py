from datetime import datetime
import json
import math
from pathlib import Path

from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from nni.nas.evaluator.pytorch import DataLoader
from nni.nas.experiment import NasExperiment

from datasets import prepare_cifar_dataset, prepare_imagenet_dataset, prepare_imagenet_100_dataset, random_split_for_search
from fbnet.lightning import FBNetClassification
from fbnet.model import FBNetSpace
from fbnet.strategy import FBNetStrategy
from fbnet.optimizer import SGDMomentum


def fbnet_search():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    id = formatted_datetime

    # Prepare datasets
    # train_data, valid_data, num_classes = prepare_cifar_dataset(nni_trace=True)
    # FBNet searches architectures by training ImageNet-100 dataset
    train_data, valid_data, num_classes = prepare_imagenet_100_dataset(nni_trace=True)
    search_train_data, search_valid_data = random_split_for_search(train_data, 0.8)

    # NAS experiment needs 3 steps
    # Step 1. Define model space
    model_space = FBNetSpace(num_classes)

    # Step 2. Define evaluator
    fast_dev_run = False

    num_gpus = 4
    batch_size_per_gpu = 120  # This should fit in RTX 3090 (22GB VRAM used)
    batch_size = batch_size_per_gpu * num_gpus  # 192 = 48 * 4

    # max_epochs = 90  # FBNet default
    # Change this for faster training (architecture_warmup_epochs stays same at 10)
    max_epochs = 45

    # FBNet paper uses two optimizers to train supernet.
    # When batch_size = 192,
    # 1. Supernet model (weight_optimizer): SGDMomentum(learning_rate=1e-1, weight_decay=1e-4)
    # 2. Softmax sampler (architecture_optimizer): Adam(learning_rate=1e-2, weight_decay=1e-4)
    scale = batch_size / 192  # Using linear scaling rule
    learning_rate = 1e-1 * scale  # Using linear scaling rule
    weight_decay = 1e-4

    architecture_learning_rate = 1e-2 * scale  # Using linear scaling rule
    architecture_weight_decay = 1e-4
    
    # ImageNet gradual warmup learning rate convention:
    #   Warm up 5 epochs per 120 epochs training
    gradual_warmup_epochs = 5 * (max_epochs / 120)

    # Set TensorBoard log and model checkpoint directory
    experiments_dir = "experiments"
    experiment_name = "search-imagenet"
    experiment_version = f"batch={batch_size}"

    # TODO(donghyeon): handle added version suffix cases (ex: "last_v1.ckpt")
    # Set auto_resume=True to resume stopped experiments when checkpoint file exists
    auto_resume = True
    resume_ckpt_filename = "last.ckpt" # (or "epoch=0-step=540.ckpt")
    checkpoint_dirpath = f"{experiments_dir}/{experiment_name}/{experiment_version}"
    checkpoint_file_path = f"{checkpoint_dirpath}/{resume_ckpt_filename}"
    # Expected checkpoint file path: ./experiments/search_imagenet/batch=192/last.ckpt
    resume = Path(checkpoint_file_path).exists() if auto_resume else False

    # FBNet paper splits train_data into 80% to train model parameters
    # and 20% to train softmax parameters
    search_train_loader = DataLoader(search_train_data, batch_size=batch_size_per_gpu, num_workers=4,
                                     pin_memory=True, persistent_workers=True)
    search_valid_loader = DataLoader(search_valid_data, batch_size=batch_size_per_gpu, num_workers=4,
                                     pin_memory=True, persistent_workers=True)
                                     

    # LightningModule is changed by FBNetStrategy.configure_oneshot_module
    evaluator = FBNetClassification(
        num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer=SGDMomentum,
        gradual_warmup_epochs = gradual_warmup_epochs,
        lr_scheduler_fn=lambda optimizer: CosineAnnealingLR(
            optimizer=optimizer,
            T_max=max_epochs,
        ),
        train_dataloaders=search_train_loader,
        val_dataloaders=search_valid_loader,
        trainer_kwargs=dict(
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=num_gpus,
            strategy=DDPStrategy(find_unused_parameters=True),
            precision="16-mixed",
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
                ModelCheckpoint(dirpath=checkpoint_dirpath, save_last=True),
            ],
            logger=TensorBoardLogger(experiments_dir, name=experiment_name, version=experiment_version),
            fast_dev_run=fast_dev_run,
        ),
        fit_kwargs={"ckpt_path": checkpoint_file_path} if resume else None,
    )

    # Step 3. Define strategy
    search_strategy = FBNetStrategy(
        architecture_learning_rate=architecture_learning_rate,
        architecture_weight_decay=architecture_weight_decay,
        architecture_warmup_epochs=10,
        gumbel_initial_temperature=5.0,
        gumbel_temperature_anneal_rate=math.exp(-0.045 * (90 / max_epochs)),  # FBNet default: exp(-0.045) for 90 epochs
        max_epochs=max_epochs,
    )

    # Run NAS experiment
    experiment = NasExperiment(model_space, evaluator, search_strategy, id=id)
    experiment.run()

    # TODO(donghyeon): FBNet paper samples 6 architectures
    # Make sure to save and load searched_arch.json works
    # Export the searched architecture
    exported_arch = experiment.export_top_models(formatter="dict")[0]

    # Serialize and save the searched architecture in json format
    Path(checkpoint_dirpath).mkdir(parents=True, exist_ok=True)
    searched_arch_id_json_path = f"{checkpoint_dirpath}/searched-arch-{id}.json"
    searched_arch_json_path = f"{checkpoint_dirpath}/searched-arch.json"
    with open(searched_arch_id_json_path, "w") as f:
        json.dump(exported_arch, f)
    with open(searched_arch_json_path, "w") as f:
        json.dump(exported_arch, f)


if __name__ == "__main__":
    fbnet_search()
