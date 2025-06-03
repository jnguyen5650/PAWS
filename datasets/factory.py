from .video_sr_dataset import VideoSRDataset
from utils.dist import is_distributed, is_main_process
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def build_dataloaders(config):
    scale_factor = config["model"].get("scale_factor", 4)
    batch_size = config["training"]["batch_size"]
    total_epochs = config["training"]["epochs"]

    train_dataset_config = config["dataset"]["train"]
    val_dataset_config = config["dataset"]["val"]

    degredation_config = config["degradation"]
    custom_degradation_config = config["custom_degradation"]

    train_dataset = VideoSRDataset(
        lr_dir=train_dataset_config["lr_dir"],
        hr_dir=train_dataset_config["hr_dir"],
        scale_factor=scale_factor,
        patch_size=train_dataset_config["patch_size"],
        num_frames=train_dataset_config["num_frames"],
        use_degradations=degredation_config.get("use_degradations", False),
        real_opt=degredation_config,
        use_custom_degradations=custom_degradation_config.get("use_custom_degradations", False),
        custom_opt=custom_degradation_config
    )

    val_dataset = VideoSRDataset(
        lr_dir=val_dataset_config["lr_dir"],
        hr_dir=val_dataset_config["hr_dir"],
        scale_factor=scale_factor,
        patch_size=val_dataset_config["patch_size"],
        num_frames=val_dataset_config["num_frames"],
        use_degradations=degredation_config.get("use_degradations", False),
        real_opt=degredation_config,
        use_custom_degradations=custom_degradation_config.get("use_custom_degradations", False),
        custom_opt=custom_degradation_config
    )

    if is_distributed():
        train_sampler = DistributedSampler(
            train_dataset, shuffle=config["dataloader"]["train"]["shuffle"]
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=config["dataloader"]["train"]["num_workers"],
            sampler=train_sampler,
            persistent_workers=config["dataloader"]["train"]["persistent_workers"],
            pin_memory=config["dataloader"]["train"]["pin_memory"],
            drop_last=True
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=config["dataloader"]["train"]["num_workers"],
            shuffle=config["dataloader"]["train"]["shuffle"],
            persistent_workers=config["dataloader"]["train"]["persistent_workers"],
            pin_memory=config["dataloader"]["train"]["pin_memory"],
            drop_last=True
        )
    
    if is_distributed():
        val_sampler = DistributedSampler(
            val_dataset, shuffle=config["dataloader"]["val"]["shuffle"], drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=config["dataloader"]["val"]["num_workers"],
            sampler=val_sampler,
            persistent_workers=config["dataloader"]["val"]["persistent_workers"],
            pin_memory=config["dataloader"]["val"]["pin_memory"],
        )
    else:
        val_sampler = None
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=config["dataloader"]["val"]["num_workers"],
            shuffle=config["dataloader"]["val"]["shuffle"],
            persistent_workers=config["dataloader"]["val"]["persistent_workers"],
            pin_memory=config["dataloader"]["val"]["pin_memory"]
        )

    if is_main_process():
        print("TOTAL TRAINING ITERATIONS:", total_epochs * len(train_loader))

    return train_loader, val_loader, len(train_loader), len(val_loader), train_sampler, val_sampler
