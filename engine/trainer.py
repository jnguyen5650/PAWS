import torch
from tqdm import tqdm
import torch.nn.functional as F
from metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt

from losses import get_refined_artifact_map
from utils.ema import update_ema
from utils.dist import is_main_process, get_rank


def compute_total_loss(
    model, lr, hr, losses, lambda_dict, config,
    ema_model=None, gan_enabled=False, discriminator=None
):
    """
    Returns:
        fake: model output
        real: ground truth
        total_loss: scalar loss for optimizer step
        loss_details: dict of per-loss scalars (for logging)
        loss_G_adv: GAN generator adversarial loss (0.0 if not enabled)
    """
    fake = model(lr)
    real = hr
    B, T, C, H, W = real.shape

    # GAN loss (generator)
    if gan_enabled and discriminator is not None:
        D_real = discriminator(real)
        D_fake = discriminator(fake)
        D_real_mean = torch.mean(D_real)
        D_fake_mean = torch.mean(D_fake)
        loss_G_adv = (F.softplus(-(D_fake - D_real_mean)).mean() +
                      F.softplus(D_real - D_fake_mean).mean())
    else:
        loss_G_adv = 0.0

    fake_flat = fake.view(B * T, C, H, W)
    real_flat = real.view(B * T, C, H, W)
    total_loss = 0.0
    loss_details = {}

    if "charbonnier" in losses:
        loss_val = losses["charbonnier"](fake_flat, real_flat)
        total_loss += lambda_dict["pixel"] * loss_val
        loss_details["Charbonnier"] = loss_val.item()
    if "tv" in losses:
        loss_val = losses["tv"](fake_flat, real_flat)
        total_loss += lambda_dict["tv"] * loss_val
        loss_details["TV"] = loss_val.item()
    if "perceptual" in losses:
        loss_val = losses["perceptual"](fake_flat, real_flat)
        total_loss += lambda_dict["perceptual"] * loss_val
        loss_details["Perceptual"] = loss_val.item()
    if config["losses"].get("lpips", False):
        fake_lpips = torch.clamp(fake_flat.float(), -1, 1)
        real_lpips = real_flat.float()

        loss_val = losses["lpips"](fake_lpips, real_lpips)
        total_loss += lambda_dict["lpips"] * loss_val
        loss_details["LPIPS"] = loss_val.item()
    if config["losses"].get("ldl", False):
        with torch.no_grad():
            output_ema = ema_model(lr).view(B * T, C, H, W)
        pixel_weight = get_refined_artifact_map(real_flat, fake_flat, output_ema, 7)
        loss_val = losses["ldl"](fake_flat * pixel_weight, real_flat * pixel_weight)
        total_loss += lambda_dict["ldl"] * loss_val
        loss_details["LDL"] = loss_val.item()

    if gan_enabled:
        total_loss += lambda_dict["adv"] * loss_G_adv
        loss_details["G Adv"] = loss_G_adv.item()

    return fake, real, total_loss, loss_details


def train_one_epoch(
    model, ema_model, discriminator, optimizer_G, optimizer_D,
    scheduler_G, scheduler_D, scaler,
    train_loader, losses, device, config, epoch, 
    use_amp, use_ema, gan_enabled, lambda_dict,
    fix_flow_epochs=0, match_terms=None
):
    model.train()
    if gan_enabled and discriminator is not None:
        discriminator.train()
    train_loss = 0.0
    total_epochs = config["training"]["epochs"]
    amp_device = device.type

    rank = get_rank()
    disable_bar = (rank != 0)

    train_pbar = tqdm(
        enumerate(train_loader, start=1),
        desc=f"Epoch {epoch+1}/{total_epochs} [Training]",
        total=len(train_loader), disable=disable_bar, leave=True, dynamic_ncols=True, mininterval=0.1)

    for step, (lr, hr) in train_pbar:
        lr = lr.to(device)
        hr = hr.to(device)
        B, T, C, H, W = hr.shape

        # Freeze flow network parameters for the first N Epochs
        if match_terms:
            if epoch + 1 <= fix_flow_epochs:
                for name, param in model.named_parameters():
                    if any(term in name for term in match_terms):
                        param.requires_grad_(False)
            else:
                for name, param in model.named_parameters():
                    if any(term in name for term in match_terms):
                        param.requires_grad_(True)

        optimizer_G.zero_grad()
        if gan_enabled and optimizer_D is not None:
            optimizer_D.zero_grad()

        if use_amp:
            with torch.autocast(device_type=amp_device):
                fake, real, total_loss, loss_details = compute_total_loss(
                    model, lr, hr, losses, lambda_dict, config,
                    ema_model=ema_model, gan_enabled=gan_enabled, discriminator=discriminator
                )
            scaler.scale(total_loss).backward()
            scaler.step(optimizer_G)
        else:
            fake, real, total_loss, loss_details = compute_total_loss(
                model, lr, hr, losses, lambda_dict, config,
                ema_model=ema_model, gan_enabled=gan_enabled, discriminator=discriminator
            )
            total_loss.backward()
            optimizer_G.step()

        if use_ema and ema_model is not None:
            src = model.module if hasattr(model,'module') else model
            dst = ema_model.module if hasattr(ema_model,'module') else ema_model
            update_ema(dst, src, alpha=config["training"].get("ema_decay", 0.999))

        if gan_enabled and (discriminator is not None) and (optimizer_D is not None):
            fake_detach = fake.detach()
            if use_amp:
                with torch.autocast(device_type=amp_device):
                    D_real = discriminator(real)
                    D_fake = discriminator(fake_detach)
                    D_real_mean = torch.mean(D_real)
                    D_fake_mean = torch.mean(D_fake)
                    loss_D = (F.softplus(-(D_real - D_fake_mean)).mean() +
                              F.softplus(D_fake - D_real_mean).mean())
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)
            else:
                D_real = discriminator(real)
                D_fake = discriminator(fake_detach)
                D_real_mean = torch.mean(D_real)
                D_fake_mean = torch.mean(D_fake)
                loss_D = (F.softplus(-(D_real - D_fake_mean)).mean() +
                          F.softplus(D_fake - D_real_mean).mean())
                loss_D.backward()
                optimizer_D.step()
            loss_details["D Adv"] = loss_D.item()
        
        if use_amp:
            scaler.update()

        scheduler_G.step()
        if gan_enabled and scheduler_D is not None:
            scheduler_D.step()

        train_loss += total_loss.item() * lr.size(0)
        postfix = {k: f"{v:.4f}" for k, v in loss_details.items()}
        postfix["Main LR"] = f"{optimizer_G.param_groups[0]['lr']:.2e}"
        postfix["Flow LR"] = f"{optimizer_G.param_groups[1]['lr']:.2e}"
        if is_main_process():
            train_pbar.set_postfix(postfix)
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    if is_main_process():
        tqdm.write(f"Epoch {epoch+1}/{total_epochs} | Train Loss: {avg_train_loss:.4f}")

    return avg_train_loss, loss_details


def validate_one_epoch(
    model, ema_model, val_loader, losses, device, config, epoch, lambda_dict,
    use_ema=False
):
    model.eval()
    val_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_imgs = 0
    sample_ema = None
    sample_fake = None
    sample_real = None

    total_epochs = config["training"]["epochs"]

    rank = get_rank()
    disable_bar = (rank != 0)

    val_pbar = tqdm(enumerate(val_loader, start=1), 
                    desc=f"Epoch {epoch+1}/{total_epochs} [Validation]", 
                    total=len(val_loader),
                    disable=disable_bar, 
                    leave=True, 
                    dynamic_ncols=True, 
                    mininterval=0.1)
    with torch.no_grad():
        for step, (lr, hr) in val_pbar:
            lr = lr.to(device)
            hr = hr.to(device)
            B, T, C, H, W = hr.shape
            fake = model(lr)
            fake = fake.view(B * T, C, H, W)
            real = hr.view(B * T, C, H, W)

            val_loss_total = 0.0
            if "charbonnier" in losses:
                val_loss_total += lambda_dict["pixel"] * losses["charbonnier"](fake, real)
            if "tv" in losses:
                val_loss_total += lambda_dict["tv"] * losses["tv"](fake, real)
            if "perceptual" in losses:
                val_loss_total += lambda_dict["perceptual"] * losses["perceptual"](fake, real)
            if config["losses"].get("lpips", False):
                fake_lpips = torch.clamp(fake.float(), -1, 1)
                real_lpips = real.float()
                val_loss_total += lambda_dict["lpips"] * losses["lpips"](fake_lpips, real_lpips)

            val_loss += val_loss_total.item() * lr.size(0)

            psnr_batch = calculate_psnr_pt(fake, real, crop_border=0, test_y_channel=False)
            ssim_batch = calculate_ssim_pt(fake, real, crop_border=0, test_y_channel=False)

            total_psnr += psnr_batch.mean().item() * fake.size(0)
            total_ssim += ssim_batch.mean().item() * fake.size(0)
            total_imgs += fake.size(0)
            
            val_pbar.set_postfix({"Val Loss": f"{val_loss_total.item():.4f}"})

            if is_main_process() and config["logging"]["enabled"]:
                if step == 1:
                    if use_ema:
                        ema_output = ema_model(lr)
                        sample_ema = ema_output[0]

                    fake_reshaped = fake.view(B, T, C, H, W)
                    sample_fake = fake_reshaped[0]
                    
                    real_reshaped = real.view(B, T, C, H, W)
                    sample_real = real_reshaped[0]

    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_psnr = total_psnr / total_imgs if total_imgs > 0 else 0.0
    avg_ssim = total_ssim / total_imgs if total_imgs > 0 else 0.0
    if is_main_process():
        tqdm.write(f"Epoch {epoch+1}/{total_epochs} | Val Loss: {avg_val_loss:.4f} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")
    
    return avg_val_loss, avg_psnr, avg_ssim, sample_fake, sample_real, sample_ema
