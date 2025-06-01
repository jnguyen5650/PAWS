import os
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

def setup_tensorboard(config):
    """
    Create a unique TensorBoard log directory and return a SummaryWriter.

    Args:
        config (dict): Config dictionary.

    Returns:
        SummaryWriter: TensorBoard writer object (or None if logging is disabled)
    """
    logging_cfg = config.get("logging", {})
    if not logging_cfg.get("enabled", False):
        return None

    experiment_name = logging_cfg.get("name", "default_experiment")
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", f"{experiment_name}_{current_datetime}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer


def log_training_scalars(tensorboard_writer, avg_train_loss, avg_val_loss, optimizer_G, avg_psnr, avg_ssim, loss_details, epoch):
    tensorboard_writer.add_scalar("Loss/Train", avg_train_loss, epoch+1)
    tensorboard_writer.add_scalar("Loss/Validation", avg_val_loss, epoch+1)
    tensorboard_writer.add_scalar("Learning Rate/Main", optimizer_G.param_groups[0]['lr'], epoch+1)
    tensorboard_writer.add_scalar("Learning Rate/Flow", optimizer_G.param_groups[1]['lr'], epoch+1)
    tensorboard_writer.add_scalar("Metrics/PSNR", avg_psnr, epoch+1)
    tensorboard_writer.add_scalar("Metrics/SSIM", avg_ssim, epoch+1)
    for key, value in loss_details.items():
        tensorboard_writer.add_scalar(f"Loss/{key}", value, epoch+1)

    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / 1024**3
        vram_reserved = torch.cuda.memory_reserved() / 1024**3
        vram_max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        vram_max_reserved = torch.cuda.max_memory_reserved() / 1024**3
        tensorboard_writer.add_scalar("GPU/VRAM_Allocated_GiB", vram_allocated, epoch+1)
        tensorboard_writer.add_scalar("GPU/VRAM_Reserved_GiB", vram_reserved, epoch+1)
        tensorboard_writer.add_scalar("GPU/Max_VRAM_Allocated_GiB", vram_max_allocated, epoch+1)
        tensorboard_writer.add_scalar("GPU/Max_VRAM_Reserved_GiB", vram_max_reserved, epoch+1)


def log_validation_images(tensorboard_writer, fake_output, real_output, ema_output=None, epoch=None):
    """
    Log fake/real (and optionally EMA) images for validation to TensorBoard.

    Args:
        tensorboard_writer: TensorBoard SummaryWriter.
        fake_output: (T, C, H, W)
        real_output: (T, C, H, W)
        ema_output: (T, C, H, W), optional
        epoch: (int) Current epoch for logging.
    """
    # Create a grid with one row (all frames in one row).
    grid_fake = vutils.make_grid(fake_output, nrow=fake_output.shape[0], normalize=True, scale_each=True)
    tensorboard_writer.add_image('Validation/Fake_Frames', grid_fake, global_step=epoch+1)
    
    grid_real = vutils.make_grid(real_output, nrow=real_output.shape[0], normalize=True, scale_each=True)
    tensorboard_writer.add_image('Validation/Real_Frames', grid_real, global_step=epoch+1)
    
    if ema_output is not None:
        grid_ema = vutils.make_grid(ema_output, nrow=ema_output.shape[0], normalize=True, scale_each=True)
        tensorboard_writer.add_image('Validation/EMA_Frames', grid_ema, global_step=epoch+1)
