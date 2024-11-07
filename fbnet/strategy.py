import math

import pytorch_lightning as pl

from nni.nas.evaluator.pytorch.lightning import LightningModule
from nni.nas.oneshot.pytorch.base_lightning import BaseOneShotLightningModule
from nni.nas.oneshot.pytorch.differentiable import LinearTemperatureScheduler
from nni.nas.strategy import GumbelDARTS

from fbnet.lightning import FBNetSearchLightningModule


class FBNetStrategy(GumbelDARTS):
    def __init__(self, *,
                 architecture_learning_rate: float=1e-2,
                 architecture_weight_decay: float=1e-4,
                 architecture_warmup_epochs: int=10,
                 gumbel_initial_temperature: float=5.0,
                 gumbel_temperature_anneal_rate: float=math.exp(-0.045),
                 max_epochs: int=90,
                 **kwargs):
        super().__init__(
            temperature=LinearTemperatureScheduler(
                init=gumbel_initial_temperature,
                min=gumbel_initial_temperature * gumbel_temperature_anneal_rate ** (max_epochs - 1),
            ),  # GumbelDARTS __init__
            arc_learning_rate=architecture_learning_rate,  # DARTS __init__
            warmup_epochs=architecture_warmup_epochs,  # DARTS __init__
            **kwargs,
        )
        self.arc_weight_decay = architecture_weight_decay
    
    def configure_oneshot_module(self, training_module: LightningModule) -> BaseOneShotLightningModule:
        return FBNetSearchLightningModule(
            training_module=training_module,
            temperature_scheduler=self.temperature,
            arc_learning_rate=self.arc_learning_rate,
            arc_weight_decay=self.arc_weight_decay,
            arc_warmup_epochs=self.warmup_epochs,
            gradient_clip_val=self.gradient_clip_val,  # DartsLightningModule __init__
            log_prob_every_n_step=self.log_prob_every_n_step,  # DartsLightningModule __init__
            penalty=self.penalty,  # DartsLightningModule __init__
        )
