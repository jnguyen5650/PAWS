from utils.config import load_config, get_lambda_dict
from utils.logging import setup_tensorboard, log_training_scalars, log_validation_images
from models.factory import build_generator, build_ema_model, build_discriminator
from losses.factory import build_losses
from datasets.factory import build_dataloaders
from utils.optimizer import build_optimizers
from utils.scheduler import build_schedulers
from utils.checkpoint import load_checkpoint_if_exists, maybe_save_checkpoint, save_final_models
from engine.trainer import train_one_epoch, validate_one_epoch
import torch
import argparse
from torchinfo import summary


def main():
    parser = argparse.ArgumentParser(description="Train a VSR model.")
    parser.add_argument("config", type=str, help="Path to config YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AMP_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tensorboard_writer = setup_tensorboard(config)
    
    losses = build_losses(config, DEVICE)
    
    model = build_generator(config, DEVICE)
    ema_model = build_ema_model(config, model, DEVICE)
    discriminator = build_discriminator(config, DEVICE)

    train_loader, val_loader, train_loader_len, _, _, _ = build_dataloaders(config)
    
    optimizer_G, optimizer_D = build_optimizers(config, model, discriminator)
    scheduler_G, scheduler_D = build_schedulers(config, train_loader_len, optimizer_G, optimizer_D)
    
    scaler = torch.amp.GradScaler(AMP_DEVICE)
    
    start_epoch = load_checkpoint_if_exists(
        config, model, optimizer_G, scheduler_G, scaler, ema_model, discriminator, optimizer_D, scheduler_D, DEVICE
    )
    end_epoch = config["training"]["epochs"]

    summary(model, input_size=(1, 5, 3, 64, 64))
    if config.get("gan", {}).get("enabled", False):
        summary(discriminator, input_size=(1, 5, 3, 64, 64))

    # Training loop
    for epoch in range(start_epoch, end_epoch):
        avg_train_loss, train_loss_details = train_one_epoch(
            model, ema_model, discriminator, optimizer_G, optimizer_D,
            scheduler_G, scheduler_D, scaler,
            train_loader, losses, DEVICE, config, epoch,
            use_amp=config["training"]["use_amp"],
            use_ema=config["training"].get("use_ema", False),
            gan_enabled=config.get("gan", {}).get("enabled", False),
            lambda_dict=get_lambda_dict(config),
            fix_flow_epochs=config["training"]["fix_flow_epochs"],
            match_terms=config.get("model", {}).get("flow_param_keywords", ["spynet"])
        )

        avg_val_loss, avg_psnr, avg_ssim, sample_fake, sample_real, sample_ema = validate_one_epoch(
            model, ema_model, val_loader, losses, DEVICE, config, epoch, get_lambda_dict(config),
            use_ema=config["training"].get("use_ema", False))
        
        if tensorboard_writer:
            log_training_scalars(
                tensorboard_writer, avg_train_loss, avg_val_loss, optimizer_G,
                avg_psnr, avg_ssim, train_loss_details, epoch
            )
            log_validation_images(
                tensorboard_writer, sample_fake, sample_real, sample_ema, epoch
            )
        
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
            use_amp=config["training"]["use_amp"],
            use_ema=config["training"].get("use_ema", False),
            gan_enabled=config.get("gan", {}).get("enabled", False),
        )
    
    if tensorboard_writer:
        tensorboard_writer.close()
    
    save_final_models(
        model,
        ema_model=ema_model,
        discriminator=discriminator,
        config=config
    )


if __name__ == "__main__":
    main()
