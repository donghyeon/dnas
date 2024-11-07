import random
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, Subset, random_split
import torch.distributed as dist

from torchvision.transforms import v2, InterpolationMode
from torchvision.datasets import CIFAR10, ImageNet

import nni

def random_split_for_search(dataset: Dataset, train_ratio: float, seed: int=0) -> List[Subset]:
    return random_split(dataset, [train_ratio, 1-train_ratio],
                        generator=torch.Generator().manual_seed(seed))


# TODO(ryan): Resolution should not be extended. They are originally 32x32-sized images.
def prepare_cifar_dataset(nni_trace=False) -> Tuple[Dataset, Dataset, int]:
    num_classes = 10
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    if nni_trace:
        DATA_CLASS = nni.trace(CIFAR10)
    else:
        DATA_CLASS = CIFAR10

    train_data = DATA_CLASS(root="./data", train=True, download=True, transform=transform)
    test_data = DATA_CLASS(root="./data", train=False, download=True, transform=transform)

    return train_data, test_data, num_classes


def prepare_imagenet_dataset(nni_trace=False) -> Tuple[Dataset, Dataset, int]:
    num_classes = 1000
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = v2.Compose([
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(),
        v2.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        v2.RandomErasing(0.1),
    ])
    val_transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if nni_trace:
        DATA_CLASS = nni.trace(ImageNet)
    else:
        DATA_CLASS = ImageNet

    train_data = DATA_CLASS(root='/datasets', split='train', transform=train_transform)
    val_data = DATA_CLASS(root='/datasets', split='val', transform=val_transform)

    return train_data, val_data, num_classes


def prepare_imagenet_tta_valid_dataset():
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    tta_transform = v2.Compose([
        v2.Resize(256),
        v2.TenCrop(224),
        v2.Lambda(lambda crops: torch.stack([v2.ToImage()(crop) for crop in crops])),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return ImageNet(root='/datasets', split='val', transform=tta_transform)


class ImageNet100(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.dataset = ImageNet(root=root, split=split)
        self.transform = transform
        
        # 모든 프로세스에서 동일한 시드 설정
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        
        if rank == 0:
            # 랭크 0에서만 랜덤 샘플링 수행
            all_classes = list(range(1000))
            random.seed(0)  # 고정된 시드 사용
            self.selected_classes = random.sample(all_classes, 100)
        else:
            self.selected_classes = [0] * 100  # 다른 랭크에서는 임시로 초기화

        if world_size > 1:
            # 선택된 클래스를 모든 프로세스에 브로드캐스트
            self.selected_classes = torch.tensor(self.selected_classes, dtype=torch.long).cuda()
            dist.broadcast(self.selected_classes, src=0)
            self.selected_classes = self.selected_classes.tolist()
        
        # 선택된 클래스에 해당하는 인덱스만 필터링
        self.indices = [i for i, (_, cls) in enumerate(self.dataset.imgs) if cls in self.selected_classes]
        
        # 클래스 라벨 매핑 (0부터 99까지)
        self.class_mapping = {old: new for new, old in enumerate(self.selected_classes)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, cls = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, self.class_mapping[cls]


def prepare_imagenet_100_dataset(nni_trace=False) -> Tuple[Dataset, Dataset, int]:
    num_classes = 100
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = v2.Compose([
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if nni_trace:
        DATA_CLASS = nni.trace(ImageNet100)
    else:
        DATA_CLASS = ImageNet100

    train_data = DATA_CLASS(root='/datasets', split='train', transform=train_transform)
    val_data = DATA_CLASS(root='/datasets', split='val', transform=val_transform)

    return train_data, val_data, num_classes
