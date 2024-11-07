import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torchvision.transforms import v2

from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer

from torchmetrics import Accuracy

from mobile_cv.model_zoo.models.fbnet_v2 import FBNet, _load_pretrained_weight


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups


NAME_MAPPING = {
    # external name : internal name
    "FBNet_a": "fbnet_a",
    "FBNet_b": "fbnet_b",
    "FBNet_c": "fbnet_c",
    "MobileNetV3": "mnv3",
    "FBNetV2_F5": "FBNetV2_L2",
}


class ExponentialMovingAverage(optim.swa_utils.AveragedModel):
    def __init__(self, model, decay):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param
        
        super().__init__(model, avg_fn=ema_avg, use_buffers=True)


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.

    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class MobileVisionFBNet(FBNet):
    def __init__(self, arch_name,
                 dim_in=3, num_classes=1000, stage_indices=None,
                 overwrite_options: Optional[List[Dict[str, int]]] = None,
                 from_public_pretrained: bool = False):
        if isinstance(arch_name, str) and arch_name in NAME_MAPPING:
            arch_name = NAME_MAPPING[arch_name]

        super().__init__(
            arch_name,
            dim_in=dim_in,
            num_classes=num_classes,
            stage_indices=stage_indices,
            overwrite_options=overwrite_options
        )

        if from_public_pretrained:
            _load_pretrained_weight(arch_name, self, True)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x)
        y = self.dropout(y)
        y = self.head(y)
        return y


class MobileVisionLightningModule(LightningModule):
    def __init__(self, model: nn.Module, num_classes: int,
                 learning_rate: float = 0.1, momentum: float = 0.9, weight_decay: float = 2e-5, norm_weight_decay: float = 0.0,
                 max_epochs: int = 600, gradual_warmup_epochs: int = 5,
                 mixup_alpha: Optional[float] = 0.2, cutmix_alpha: Optional[float] = 1.0,
                 label_smoothing: float = 0.1,
                 model_ema: bool = True, model_ema_steps: int = 32, model_ema_decay: float = 0.99998,
                 batch_size: int = 256,
                 ):
        super().__init__()

        self.model = model
        self.model_ema = model_ema
        if model_ema:
            self.model_ema_steps = model_ema_steps
            self.model_ema_decay = model_ema_decay
            adjust = batch_size * model_ema_steps / max_epochs
            alpha = 1.0 - model_ema_decay
            alpha = min(1.0, alpha * adjust)
            self.model_ema = ExponentialMovingAverage(model, decay=1.0 - alpha)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.base_learning_rate = learning_rate
        parameters = set_weight_decay(model=model, weight_decay=weight_decay, norm_weight_decay=norm_weight_decay)
        self.optimizer = optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        self.max_epochs = max_epochs
        # milestone = int(max_epochs * 3/4)
        # factor = 1e-2
        # scheduler1 = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=milestone, eta_min=(factor * learning_rate))
        # scheduler2 = lr_scheduler.ConstantLR(self.optimizer, total_iters=(max_epochs - milestone), factor=factor)
        # self.scheduler = lr_scheduler.SequentialLR(self.optimizer, schedulers=[scheduler1, scheduler2], milestones=[milestone])
        # min_lr = 0.01
        # self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.01**(1/(max_epochs-1)))
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)

        self.gradual_warmup_epochs = gradual_warmup_epochs

        # Mix Augmnetations
        mix_augs = []
        if mixup_alpha and mixup_alpha > 0:
            mix_augs.append(v2.MixUp(alpha=mixup_alpha, num_classes=num_classes))
        if cutmix_alpha and cutmix_alpha > 0:
            mix_augs.append(v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes))
        self.mixup_or_cutmix = v2.RandomChoice(mix_augs) if mix_augs else None

        self.train_acc1 = Accuracy(task="multiclass", num_classes=num_classes, average="micro", multidim_average="global", top_k=1)
        self.train_acc5 = Accuracy(task="multiclass", num_classes=num_classes, average="micro", multidim_average="global", top_k=5)
        self.valid_acc1 = Accuracy(task="multiclass", num_classes=num_classes, average="micro", multidim_average="global", top_k=1)
        self.valid_acc5 = Accuracy(task="multiclass", num_classes=num_classes, average="micro", multidim_average="global", top_k=5)
        self.test_acc1 = Accuracy(task="multiclass", num_classes=num_classes, average="micro", multidim_average="global", top_k=1)
        self.test_acc5 = Accuracy(task="multiclass", num_classes=num_classes, average="micro", multidim_average="global", top_k=5)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        original_labels = labels

        if self.mixup_or_cutmix:
            images, labels = self.mixup_or_cutmix(images, labels)

        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        self.train_acc1(outputs, original_labels)
        self.train_acc5(outputs, original_labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        self.log('train_acc1', self.train_acc1, prog_bar=True, on_step=True)
        self.log('train_acc5', self.train_acc5, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        model_evaluated = self.model_ema if self.model_ema else self.model
        outputs = model_evaluated(images)

        loss = self.criterion(outputs, labels)
        self.valid_acc1(outputs, labels)
        self.valid_acc5(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc1', self.valid_acc1, prog_bar=True, on_epoch=True)
        self.log('val_acc5', self.valid_acc5, on_epoch=True)

    def test_step(self, batch, batch_idx):
        tta_images, labels = batch
        bs, ncrops, c, h, w = tta_images.size()

        model_evaluated = self.model_ema if self.model_ema else self.model
        outputs = model_evaluated(tta_images.view(-1, c, h, w)) # fuse batch size and ncrops
        results = F.softmax(outputs, dim=1)
        results_avg = results.view(bs, ncrops, -1).mean(1) # avg over crops

        self.test_acc1(results_avg, labels)
        self.test_acc5(results_avg, labels)
        self.log('test_acc1', self.test_acc1, on_epoch=True)
        self.log('test_acc5', self.test_acc5, on_epoch=True)

    def optimizer_step(
            self, epoch: int, batch_idx: int,
            optimizer: Union[optim.Optimizer, LightningOptimizer],
            optimizer_closure: Optional[Callable[[], Any]] = None,
            ) -> None:
        
        if self.gradual_warmup_epochs:
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

        if self.model_ema and batch_idx % self.model_ema_steps == 0:
            self.model_ema.update_parameters(self.model)
            if self.gradual_warmup_epochs and epoch < self.gradual_warmup_epochs:
                self.model_ema.n_averaged.fill_(0)  # Reset EMA buffer

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
