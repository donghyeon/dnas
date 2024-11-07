from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer

import nni
from nni.nas.evaluator.pytorch import ClassificationModule, Lightning, DataLoader, Trainer
from nni.nas.oneshot.pytorch.differentiable import GumbelDartsLightningModule, LinearTemperatureScheduler


# Only used for FBNet search stage
# Extend GumbelDartsLightningModule to support weight decay in architecture optimizer
@nni.trace
class FBNetSearchLightningModule(GumbelDartsLightningModule):
    def __init__(self, training_module: pl.LightningModule,
                 temperature_scheduler: LinearTemperatureScheduler,
                 arc_learning_rate: float,
                 arc_weight_decay: float,
                 arc_warmup_epochs: int,
                 **kwargs):
        super().__init__(training_module,
                         temperature_scheduler,
                         arc_learning_rate=arc_learning_rate,
                         warmup_epochs=arc_warmup_epochs,
                         **kwargs)
        self.arc_weight_decay = arc_weight_decay
    
    def configure_architecture_optimizers(self):
        ctrl_params = self.arch_parameters()
        if not ctrl_params:
            raise ValueError('No architecture parameters found. Nothing to search.')
        ctrl_optim = optim.Adam(ctrl_params,
                                lr=self.arc_learning_rate,
                                weight_decay=self.arc_weight_decay)
        return ctrl_optim


# Extend Classification evaluator to support learning rate scheduler and resume training
@nni.trace
class FBNetClassification(Lightning):
    def __init__(
            self,
                 num_classes: Optional[int] = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 gradual_warmup_epochs: float = 5.0,
                 lr_scheduler_fn: Optional[Callable[[optim.Optimizer],
                                                    optim.lr_scheduler.LRScheduler]] = None,
                 train_dataloaders: Optional[DataLoader] = None,
                 val_dataloaders: Union[DataLoader, List[DataLoader], None] = None,
                 fit_kwargs: Optional[Dict[str, Any]] = None,
                 trainer_kwargs: Optional[Dict[str, Any]] = None):
        module = ClassificationModuleWithLRScheduler(
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradual_warmup_epochs=gradual_warmup_epochs,
            optimizer=optimizer,
            lr_scheduler_fn=lr_scheduler_fn,
        )
        trainer_kwargs = trainer_kwargs or {}
        super().__init__(module, Trainer(**trainer_kwargs),
                         train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders,
                         fit_kwargs=fit_kwargs)


# Extend ClassificationModule to support learning rate scheduler
@nni.trace
class ClassificationModuleWithLRScheduler(ClassificationModule):
    def __init__(self,
                 learning_rate: float = 0.001,
                 gradual_warmup_epochs: float = 5.0,
                 lr_scheduler_fn: Optional[Callable[[optim.Optimizer],
                                                    optim.lr_scheduler.LRScheduler]] = None,
                 **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.base_learning_rate = learning_rate
        self.gradual_warmup_epochs = gradual_warmup_epochs
        self.lr_scheduler_fn = lr_scheduler_fn

    # Set learning rate scheduler
    def configure_optimizers(self):
        optimizer = super().configure_optimizers()
        if self.lr_scheduler_fn:
            lr_scheduler = self.lr_scheduler_fn(optimizer)
            return [optimizer], [lr_scheduler]
        return optimizer
    
    # Set gradual warmup learning rate at initial steps
    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Union[optim.Optimizer, LightningOptimizer],
            optimizer_closure: Optional[Callable[[], Any]] = None,
            ) -> None:
        
        # Gradual warmup schedule without optim.lr_scheduler.LRScheduler
        warmup_epochs = self.gradual_warmup_epochs
        steps_per_epoch = self.trainer.num_training_batches

        current_step = epoch * steps_per_epoch + batch_idx
        warmup_steps = warmup_epochs * steps_per_epoch

        if current_step < warmup_steps:
            lr_scale = (current_step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.base_learning_rate

        # update params
        optimizer.step(closure=optimizer_closure)
