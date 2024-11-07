import torch.optim as optim

class SGDMomentum(optim.SGD):
    def __init__(self, params, lr, momentum=0.9, **kwargs):
        super().__init__(params, lr, momentum, **kwargs)
