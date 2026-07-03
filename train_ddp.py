from utils.traceback import *  # noqa: F401, F403
from utils.dist import is_main_process
from utils.config import load_config, get_lambda_dict
from utils.logging import setup_tensorboard, log_training_scalars, log_validation_images
from models.factory import build_generator, build_ema_model, build_discriminator
from losses.factory import build_losses
from datasets.factory import build_dataloaders, apply_resume_skip_to_train_loader
from utils.optimizer import build_optimizers
from utils.scheduler import build_schedulers
from utils.checkpoint import load_checkpoint_if_exists, maybe_save_checkpoint, save_final_models
from engine.trainer import train_one_epoch, validate_one_epoch
import torch
import torch.distributed as dist
import argparse
from torchinfo import summary


def main():
    parser = argparse.ArgumentParser(description="Train a VSR model.")
    parser.add_argument("config", type=str, help="Path to config YAML file")
    parser.add_argument(
        "--local-rank", "--local_rank", type=int, default=0,
        help="DDP local process rank (passed automatically with torchrun)"
    )
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="gloo") # nccl
    torch.cuda.set_device(args.local_rank)

    config = load_config(args.config)
    DEVICE = torch.device(f"cuda:{args.local_rank}")
    AMP_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if is_main_process():
        tensorboard_writer = setup_tensorboard(config)
    
    losses = build_losses(config, DEVICE)
    
    model = build_generator(config, DEVICE)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], find_unused_parameters=True
    )
    ema_model = build_ema_model(config, model, DEVICE)

    discriminator = build_discriminator(config, DEVICE)
    if discriminator is not None:
        discriminator = torch.nn.parallel.DistributedDataParallel(
            discriminator, device_ids=[args.local_rank], find_unused_parameters=True
        )

    train_loader, val_loader, train_loader_len, val_loader_len, train_sampler, val_sampler = build_dataloaders(config)
    
    optimizer_G, optimizer_D = build_optimizers(config, model, discriminator)
    scheduler_G, scheduler_D = build_schedulers(config, train_loader_len, optimizer_G, optimizer_D)
    
    scaler = torch.amp.GradScaler(AMP_DEVICE)
    
    start_epoch, global_step, step_in_epoch = load_checkpoint_if_exists(
        config, model, optimizer_G, scheduler_G, scaler, ema_model, discriminator, optimizer_D, scheduler_D, DEVICE
    )
    end_epoch = config["training"]["epochs"]

    # If checkpoint is at/after the end of the epoch, start next epoch instead
    if step_in_epoch >= train_loader_len:
        start_epoch += 1
        step_in_epoch = 0

    train_loader = apply_resume_skip_to_train_loader(train_loader, step_in_epoch)

    if is_main_process():
        print("\n" + "=" * 72)
        print("Model Summary | Generator (G)")
        print("=" * 72)
        summary(model, input_size=(1, 5, 3, 64, 64))
        print()
        if config.get("gan", {}).get("enabled", False):
            print("\n" + "=" * 72)
            print("Model Summary | Discriminator (D)")
            print("=" * 72)
            summary(discriminator, input_size=(1, 5, 3, 64, 64))
            print()

    # Training loop
    for epoch in range(start_epoch, end_epoch):
        # Make sure each epoch the sampler shuffles differently
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        avg_train_loss, train_loss_details, global_step = train_one_epoch(
            model,
            ema_model,
            discriminator,
            optimizer_G,
            optimizer_D,
            scheduler_G,
            scheduler_D,
            scaler,
            train_loader,
            losses,
            DEVICE,
            config,
            epoch,
            global_step,
            step_in_epoch,
            lambda_dict=get_lambda_dict(config)
        )
        
        step_in_epoch = 0

        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        avg_val_loss, avg_psnr, avg_ssim, sample_input, sample_fake, sample_real, sample_ema, sample_clean = validate_one_epoch(
            model, ema_model, val_loader, losses, DEVICE, config, epoch, get_lambda_dict(config))
        

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        local_train_loss = torch.tensor(avg_train_loss, device=DEVICE)
        local_val_loss   = torch.tensor(avg_val_loss,   device=DEVICE)
        local_psnr       = torch.tensor(avg_psnr,       device=DEVICE)
        local_ssim       = torch.tensor(avg_ssim,       device=DEVICE)

        dist.all_reduce(local_train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_val_loss,   op=dist.ReduceOp.SUM)
        dist.all_reduce(local_psnr,       op=dist.ReduceOp.SUM)
        dist.all_reduce(local_ssim,       op=dist.ReduceOp.SUM)

        global_train_loss = (local_train_loss / world_size).item()
        global_val_loss   = (local_val_loss   / world_size).item()
        global_psnr       = (local_psnr       / world_size).item()
        global_ssim       = (local_ssim       / world_size).item()

        global_loss_details = {}
        for key, local_val in train_loss_details.items():
            t = torch.tensor(local_val, device=DEVICE)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            global_loss_details[key] = (t / world_size).item()

        if is_main_process() and tensorboard_writer is not None:
            log_training_scalars(
                tensorboard_writer,
                global_train_loss,
                global_val_loss,
                optimizer_G,
                global_psnr,
                global_ssim,
                global_loss_details,
                epoch
            )
            log_validation_images(
                tensorboard_writer, sample_input, sample_fake, sample_real, sample_ema, sample_clean, epoch
            )
        
        if is_main_process():
            maybe_save_checkpoint(
                epoch=epoch,
                config=config,
                model=model,
                optimizer_G=optimizer_G,
                scheduler_G=scheduler_G,
                scaler=scaler,
                ema_model=ema_model,
                discriminator=discriminator,
                optimizer_D=optimizer_D,
                scheduler_D=scheduler_D,
                global_step=global_step,
                step_in_epoch=step_in_epoch
            )
    
    if is_main_process() and tensorboard_writer:
        tensorboard_writer.close()
    
    if is_main_process():
        save_final_models(
            model,
            ema_model=ema_model,
            discriminator=discriminator,
            config=config
        )


if __name__ == "__main__":
    main()
