# data/build.py
# ------------------------------------------------------------
# Dataset and dataloader construction for pollen recognition.
#
# Supported datasets:
#   1. Pollen149
#   2. New Zealand Pollen Dataset
#   3. CPD-1
#
# ------------------------------------------------------------

import os
import torch
import torch.distributed as dist

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup, create_transform

from .pollen149 import Pollen149CSVDataset
from .new_zealand_pollen import NewZealandPollenCSVDataset
from .cpd1 import CPD1CSVDataset


try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == "bicubic":
            return InterpolationMode.BICUBIC
        elif method == "lanczos":
            return InterpolationMode.LANCZOS
        elif method == "hamming":
            return InterpolationMode.HAMMING
        else:
            return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms
    timm_transforms._pil_interp = _pil_interp

except Exception:
    from timm.data.transforms import _pil_interp


def build_loader(config):
    """
    Build train / validation / test dataloaders.
    """

    if dist.is_available() and dist.is_initialized():
        global_rank = dist.get_rank()
        num_tasks = dist.get_world_size()
    else:
        global_rank = int(os.environ.get("RANK", -1))
        num_tasks = int(os.environ.get("WORLD_SIZE", 1))

    config.defrost()

    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(
        split="train",
        config=config
    )
    dataset_val, _ = build_dataset(
        split="val",
        config=config
    )
    dataset_test, _ = build_dataset(
        split="test",
        config=config
    )

    config.freeze()

    if num_tasks > 1:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    if num_tasks > 1 and not config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val,
            shuffle=config.TEST.SHUFFLE
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    if num_tasks > 1 and not config.TEST.SEQUENTIAL:
        sampler_test = torch.utils.data.distributed.DistributedSampler(
            dataset_test,
            shuffle=False
        )
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    mixup_fn = None

    mixup_active = (
        config.AUG.MIXUP > 0
        or config.AUG.CUTMIX > 0.0
        or config.AUG.CUTMIX_MINMAX is not None
    )

    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES
        )

    return (
        dataset_train,
        dataset_val,
        dataset_test,
        data_loader_train,
        data_loader_val,
        data_loader_test,
        mixup_fn
    )


def build_dataset(split, config):
    """
    Build a dataset according to config.DATA.DATASET.

    Supported dataset names:
        - pollen149
        - new_zealand_pollen
        - NZ_pollen
        - cpd1
    """

    is_train = split == "train"

    if split == "train":
        transform = build_transform(True, config)
    elif split in ["val", "test"]:
        transform = build_transform(False, config)
    else:
        raise ValueError(f"Unknown split: {split}")

    if config.DATA.DATASET == "pollen149":
        csv_name = _resolve_csv_name(
            split=split,
            train_csv="train.csv",
            val_csv="val.csv",
            test_csv="test.csv"
        )

        csv_path = os.path.join(config.DATA.DATA_PATH, csv_name)
        _check_csv_exists(csv_path, split)

        dataset = Pollen149CSVDataset(
            root=config.DATA.DATA_PATH,
            csv_path=csv_path,
            transform=transform,
            is_train=is_train
        )

        num_classes = 149
        return dataset, num_classes

    elif config.DATA.DATASET in ["new_zealand_pollen", "NZ_pollen"]:
        csv_name = _resolve_csv_name(
            split=split,
            train_csv="train.csv",
            val_csv="val.csv",
            test_csv="test.csv"
        )

        csv_path = os.path.join(config.DATA.DATA_PATH, csv_name)
        _check_csv_exists(csv_path, split)

        dataset = NewZealandPollenCSVDataset(
            root=config.DATA.DATA_PATH,
            csv_path=csv_path,
            transform=transform,
            is_train=is_train
        )

        num_classes = 46
        return dataset, num_classes

    elif config.DATA.DATASET == "cpd1":
        csv_name = _resolve_csv_name(
            split=split,
            train_csv="train.csv",
            val_csv="val.csv",
            test_csv="test.csv"
        )

        csv_path = os.path.join(config.DATA.DATA_PATH, csv_name)
        _check_csv_exists(csv_path, split)

        dataset = CPD1CSVDataset(
            root=config.DATA.DATA_PATH,
            csv_path=csv_path,
            transform=transform,
            is_train=is_train
        )

        num_classes = 20
        return dataset, num_classes

    else:
        raise NotImplementedError(
            "Unsupported dataset: "
            f"{config.DATA.DATASET}. "
            "Supported datasets are: pollen149, new_zealand_pollen, "
            "NZ_pollen, and cpd1."
        )


def _resolve_csv_name(split, train_csv, val_csv, test_csv):
    if split == "train":
        return train_csv
    elif split == "val":
        return val_csv
    elif split == "test":
        return test_csv
    else:
        raise ValueError(f"Unknown split: {split}")


def _check_csv_exists(csv_path, split):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file for split '{split}' not found: {csv_path}"
        )


def build_transform(is_train, config):
    """
    Build image transformation pipeline.

    Training:
        Uses timm create_transform.

    Validation / Test:
        Resize or center crop according to config.TEST.CROP,
        then normalize using ImageNet mean and std.
    """

    resize_im = config.DATA.IMG_SIZE > 32

    if is_train:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=(
                config.AUG.COLOR_JITTER
                if config.AUG.COLOR_JITTER > 0
                else None
            ),
            auto_augment=(
                config.AUG.AUTO_AUGMENT
                if config.AUG.AUTO_AUGMENT != "none"
                else None
            ),
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )

        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                config.DATA.IMG_SIZE,
                padding=4
            )

        return transform

    transform_list = []

    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)

            transform_list.append(
                transforms.Resize(
                    size,
                    interpolation=_pil_interp(config.DATA.INTERPOLATION)
                )
            )
            transform_list.append(
                transforms.CenterCrop(config.DATA.IMG_SIZE)
            )
        else:
            transform_list.append(
                transforms.Resize(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=_pil_interp(config.DATA.INTERPOLATION)
                )
            )

    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize(
            IMAGENET_DEFAULT_MEAN,
            IMAGENET_DEFAULT_STD
        )
    )

    return transforms.Compose(transform_list)
