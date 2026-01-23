import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt

from losses import get_refined_artifact_map
from utils.ema import update_ema
from utils.dist import is_main_process, get_rank
from utils.logging import format_error_msg
from utils.checkpoint import maybe_save_checkpoint
from datasets.factory import SkipBatchSamplerOnce


def guard_finite(x, name, enabled=True):
    if not enabled or x is None:
        return
    if not torch.is_tensor(x):
        return
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).sum().item()
        raise RuntimeError(format_error_msg("Numerics", name, "non-finite values", bad, get_rank()))


def register_act_nan_hooks(model, discriminator=None, tag_G="G", tag_D="D"):
    rank = get_rank()
    def hook_check(name):
        def check_tensor(tag, t):
            if t is None or not torch.is_tensor(t):
                return
            bad = (~torch.isfinite(t)).sum()
            if bad.item() > 0:
                raise RuntimeError(format_error_msg("Activations", name, f"non-finite {tag}", bad.item(), rank))
        def fn(mod, inp, out):
            if isinstance(inp, (tuple, list)):
                for x in inp: check_tensor("INPUT", x)
            else:
                check_tensor("INPUT", inp)
            if isinstance(out, (tuple, list)):
                for o in out: check_tensor("OUTPUT", o)
            else:
                check_tensor("OUTPUT", out)
        return fn

    handles = []
    for n, m in model.named_modules():
        handles.append(m.register_forward_hook(hook_check(f"{tag_G}:{n}")))
    if discriminator is not None:
        for n, m in discriminator.named_modules():
            handles.append(m.register_forward_hook(hook_check(f"{tag_D}:{n}")))
    return handles


def compute_total_loss(
    fake, real, losses, lambda_dict, config,
    lr=None, ema_model=None, D_real=None, D_fake=None, amp_device='cuda',
    lqs_clean=None
):
    """
    Returns:
        fake: model output
        real: ground truth
        total_loss: scalar loss for optimizer step
        loss_details: dict of per-loss scalars (for logging)
        loss_G_adv: GAN generator adversarial loss (0.0 if not enabled)
    """
    B, T, C, H, W = real.shape

    with torch.autocast(device_type=amp_device, enabled=False):
        fake_flat = fake.float().view(B * T, C, H, W)
        real_flat = real.float().view(B * T, C, H, W)
        total_loss = torch.zeros((), device=fake.device, dtype=torch.float32)
        loss_details = {}
        if lqs_clean is not None:
            sf = int(config["model"].get("scale_factor", 4))
            # target: HR downsampled to LR with area pooling
            gt_clean = F.interpolate(
                real_flat, scale_factor=1.0 / sf, mode="area", recompute_scale_factor=False
            )
            pred_clean = lqs_clean.view(B * T, C, H // sf, W // sf).float()
            loss_details["Cleaning"] = 0.0

        if "charbonnier" in losses:
            loss_val = losses["charbonnier"](fake_flat, real_flat)
            total_loss += lambda_dict["charbonnier"] * loss_val
            loss_details["Charbonnier"] = lambda_dict["charbonnier"]* loss_val.detach().cpu().item()
        if "tv" in losses:
            loss_val = losses["tv"](fake_flat)
            total_loss += lambda_dict["tv"] * loss_val
            loss_details["TV"] = lambda_dict["tv"] * loss_val.detach().cpu().item()
        if config["losses"].get("lpips", False):
            fake_lpips = torch.clamp(fake_flat, -1, 1)
            real_lpips = real_flat
            loss_val = losses["lpips"](fake_lpips, real_lpips)
            total_loss += lambda_dict["lpips"] * loss_val
            loss_details["LPIPS"] = lambda_dict["lpips"] * loss_val.detach().cpu().item()
        if config["losses"].get("dists", False):
            fake_dists = torch.clamp((fake_flat + 1.0) * 0.5, 0.0, 1.0)
            real_dists = torch.clamp((real_flat + 1.0) * 0.5, 0.0, 1.0)
            loss_val = losses["dists"](fake_dists, real_dists)
            total_loss += lambda_dict["dists"] * loss_val
            loss_details["DISTS"] = lambda_dict["dists"] * loss_val.detach().cpu().item()
        if config["losses"].get("ldl", False):
            with torch.no_grad():
                out_ema = ema_model(lr)
                if isinstance(out_ema, (tuple, list)):
                    out_ema = out_ema[0]
                output_ema = out_ema.float().view(B * T, C, H, W)
            pixel_weight = get_refined_artifact_map(real_flat, fake_flat, output_ema, 7)
            loss_val = losses["ldl"](fake_flat * pixel_weight, real_flat * pixel_weight)
            total_loss += lambda_dict["ldl"] * loss_val
            loss_details["LDL"] = lambda_dict["ldl"] * loss_val.detach().cpu().item()
        if lqs_clean is not None and config["losses"].get("cleaning_charbonnier", False):
            loss_val = losses["cleaning_charbonnier"](pred_clean, gt_clean)
            total_loss += lambda_dict["cleaning_charbonnier"] * loss_val
            loss_details["Cleaning"] += lambda_dict["cleaning_charbonnier"] * loss_val.detach().cpu().item()
        if lqs_clean is not None and config["losses"].get("cleaning_dists", False):
            pred01 = torch.clamp((pred_clean + 1.0) * 0.5, 0.0, 1.0)
            gt01   = torch.clamp((gt_clean   + 1.0) * 0.5, 0.0, 1.0)
            loss_val = losses["cleaning_dists"](pred01, gt01)
            total_loss += lambda_dict["cleaning_dists"] * loss_val
            loss_details["Cleaning"] += lambda_dict["cleaning_dists"] * loss_val.detach().cpu().item()

        # GAN loss (generator)
        if D_real is not None and D_fake is not None:
            D_real_mean = torch.mean(D_real.float())
            D_fake_mean = torch.mean(D_fake.float())
            loss_G_adv = (F.softplus(-(D_fake - D_real_mean)).mean() +
                        F.softplus(D_real - D_fake_mean).mean())
            total_loss += lambda_dict["adv"] * loss_G_adv
            loss_details["G Adv"] = lambda_dict["adv"] * loss_G_adv.detach().cpu().item()

    return total_loss, loss_details


def train_one_epoch(
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
    device,
    config,
    epoch,
    global_step,
    step_in_epoch,
    lambda_dict
):
    use_amp = config["training"]["use_amp"]
    use_ema = config["training"].get("use_ema", False)
    gan_enabled = config.get("gan", {}).get("enabled", False)

    model.train()
    if gan_enabled and discriminator is not None:
        discriminator.train()
    train_loss = 0.0
    processed_samples = 0
    total_epochs = config["training"]["epochs"]
    amp_device = device.type

    fix_flow_epochs = config.get("training", {}).get("fix_flow_epochs", 0)
    fix_flow_iters = config.get("training", {}).get("fix_flow_iters", 0)
    match_terms = config.get("model", {}).get("flow_param_keywords", ["flow_net"])
    save_every_iters = config.get("training", {}).get("save_every_iters", None)
    save_every_iters = int(save_every_iters) if save_every_iters is not None else None

    r1_gamma = float(config.get("gan", {}).get("r1_gamma", -1.0))
    r1_every = int(config.get("gan", {}).get("r1_every", 16))

    rank = get_rank()
    disable_bar = (rank != 0)

    nan_guard = bool(config.get("debug", {}).get("nan_guard", True))
    enable_hooks = bool(config.get("debug", {}).get("activation_hooks", False))

    handles = []
    if enable_hooks:
        handles = register_act_nan_hooks(model, discriminator)

    try:
        # If we resumed and wrapped the batch_sampler, len(train_loader) is "remaining",
        # but we want the progress bar to reflect full epoch progress.
        is_wrapped = isinstance(getattr(train_loader, "batch_sampler", None), SkipBatchSamplerOnce)
        if is_wrapped:
            # Full epoch batches live in the inner batch_sampler
            epoch_total = len(train_loader.batch_sampler.batch_sampler)
            initial = int(step_in_epoch or 0)
        else:
            epoch_total = len(train_loader)
            initial = 0

        train_pbar = tqdm(
            enumerate(train_loader, start=step_in_epoch + 1),
            desc=f"Epoch {epoch+1}/{total_epochs} [Training]",
            total=epoch_total,
            initial=initial,
            disable=disable_bar,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.1
        )

        for step, (lr, hr) in train_pbar:
            global_step += 1
            lr = lr.to(device)
            hr = hr.to(device)
            B, T, C, H, W = hr.shape
            if (not torch.isfinite(lr).all()) or (not torch.isfinite(hr).all()):
                raise RuntimeError(format_error_msg("Data", "train batch", "non-finite tensors in lr/hr", rank=rank))

            # Freeze flow network parameters for the first N Epochs or first N steps
            if match_terms:
                if epoch + 1 <= fix_flow_epochs or global_step <= fix_flow_iters:
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

            D_real, D_fake = None, None
            if gan_enabled and discriminator is not None:
                for p in discriminator.parameters():
                    p.requires_grad_(False)

            if use_amp:
                with torch.autocast(device_type=amp_device):
                    want_clean = bool(config["losses"].get("cleaning_charbonnier", False)) or bool(config["losses"].get("cleaning_dists", False))
                    out = model(lr, return_lqs=True) if want_clean else model(lr)
                    if isinstance(out, (tuple, list)):
                        fake, lqs_clean = out
                    else:
                        fake, lqs_clean = out, None
                    real = hr
                    if gan_enabled and discriminator is not None:
                        domain = config.get("gan", {}).get("domain", "sr")
                        if domain == "clean":
                            assert lqs_clean is not None, "GAN domain is 'clean' but lqs_clean is None (return_lqs must be True). Was cleaning loss disabled?"

                            # build LR targets for GAN on the cleaning branch
                            sf = int(config["model"].get("scale_factor", 4))
                            B,T,C,H,W = real.shape
                            gt_clean = F.interpolate(
                                real.view(B*T, C, H, W), scale_factor=1.0/sf, mode="area", recompute_scale_factor=False
                            )                             # [B*T,3,H/sf,W/sf]
                            pred_clean = lqs_clean.view(B, T, C, H//sf, W//sf)
                            gt_clean = gt_clean.view(B, T, C, H//sf, W//sf)
                            D_real = discriminator(gt_clean)
                            D_fake = discriminator(pred_clean)
                        else:  # "sr" (default)
                            D_real = discriminator(real)
                            D_fake = discriminator(fake)
            else:
                want_clean = bool(config["losses"].get("cleaning_charbonnier", False)) or bool(config["losses"].get("cleaning_dists", False))
                out = model(lr, return_lqs=True) if want_clean else model(lr)
                if isinstance(out, (tuple, list)):
                    fake, lqs_clean = out
                else:
                    fake, lqs_clean = out, None
                real = hr
                if gan_enabled and discriminator is not None:
                    domain = config.get("gan", {}).get("domain", "sr")
                    if domain == "clean":
                        assert lqs_clean is not None, "GAN domain is 'clean' but lqs_clean is None (return_lqs must be True). Was cleaning loss disabled?"

                        # build LR targets for GAN on the cleaning branch
                        sf = int(config["model"].get("scale_factor", 4))
                        B,T,C,H,W = real.shape
                        gt_clean = F.interpolate(
                            real.view(B*T, C, H, W), scale_factor=1.0/sf, mode="area", recompute_scale_factor=False
                        )
                        pred_clean = lqs_clean.view(B, T, C, H//sf, W//sf)
                        gt_clean = gt_clean.view(B, T, C, H//sf, W//sf)
                        D_real = discriminator(gt_clean)
                        D_fake = discriminator(pred_clean)
                    else:
                        D_real = discriminator(real)
                        D_fake = discriminator(fake)
            
            total_loss, loss_details = compute_total_loss(
                        fake, real, losses, lambda_dict, config,
                        lr=lr, ema_model=ema_model, D_real=D_real, D_fake=D_fake, amp_device=amp_device,
                        lqs_clean=lqs_clean
                )
            guard_finite(total_loss, "G total_loss", nan_guard)

            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer_G)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if nan_guard and (not torch.isfinite(gnorm)):
                    raise RuntimeError(format_error_msg("GradNorm", "G", "non-finite gradient norm", rank=rank))
                scaler.step(optimizer_G)
            else:
                total_loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if nan_guard and (not torch.isfinite(gnorm)):
                    raise RuntimeError(format_error_msg("GradNorm", "G", "non-finite gradient norm", rank=rank))
                optimizer_G.step()
            
            if gan_enabled and discriminator is not None:
                for p in discriminator.parameters():
                    p.requires_grad_(True)

            if use_ema and ema_model is not None:
                src = model.module if hasattr(model,'module') else model
                dst = ema_model.module if hasattr(ema_model,'module') else ema_model
                update_ema(dst, src, alpha=config["training"].get("ema_decay", 0.999))

            if gan_enabled and (discriminator is not None) and (optimizer_D is not None):
                domain = config.get("gan", {}).get("domain", "sr")
                if domain == "clean":
                    assert lqs_clean is not None, "GAN domain is 'clean' but lqs_clean is None (return_lqs must be True). Was cleaning loss disabled?"

                    # build clean-domain tensors again (detached for D)
                    sf = int(config["model"].get("scale_factor", 4))
                    B,T,C,H,W = real.shape
                    gt_clean = F.interpolate(real.view(B*T,C,H,W), scale_factor=1.0/sf, mode="area", recompute_scale_factor=False)
                    pred_clean = lqs_clean.view(B,T,C,H//sf,W//sf).detach()
                    x_real_D = gt_clean.view(B, T, C, H//sf, W//sf)
                    x_fake_D = pred_clean
                else:
                    fake_detach = fake.detach()
                    x_real_D = real
                    x_fake_D = fake_detach
                if use_amp:
                    with torch.autocast(device_type=amp_device):
                        D_real = discriminator(x_real_D)
                        D_fake = discriminator(x_fake_D)
                else:
                    D_real = discriminator(x_real_D)
                    D_fake = discriminator(x_fake_D)
                
                with torch.autocast(device_type=amp_device, enabled=False):
                    D_real_f32 = D_real.float()
                    D_fake_f32 = D_fake.float()
                    D_real_mean = D_real_f32.mean()
                    D_fake_mean = D_fake_f32.mean()
                    loss_D = (
                        F.softplus(-(D_real_f32 - D_fake_mean)).mean()
                        + F.softplus(  D_fake_f32 - D_real_mean).mean()
                    )

                    # R1 regularization (lazy)
                    do_r1 = r1_gamma > 0.0 and global_step % r1_every == 0
                    r1_term = None
                    if do_r1:
                        real_r1 = x_real_D.detach().to(torch.float32).requires_grad_(True)
                        pred_real_r1 = discriminator(real_r1)
                        grad_real = torch.autograd.grad(
                            outputs=pred_real_r1.sum(),
                            inputs=real_r1,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True
                        )[0]
                        r1 = grad_real.pow(2).reshape(grad_real.size(0), -1).sum(1).mean()
                        r1_term = 0.5 * r1_gamma * r1 * r1_every
                        loss_D = loss_D + r1_term
                    guard_finite(loss_D, "D loss", nan_guard)
                
                if use_amp:
                    scaler.scale(loss_D).backward()
                    scaler.unscale_(optimizer_D)
                    dnorm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    if nan_guard and (not torch.isfinite(dnorm)):
                        raise RuntimeError(format_error_msg("GradNorm", "D", "non-finite gradient norm", rank=rank))
                    scaler.step(optimizer_D)
                else:
                    loss_D.backward()
                    dnorm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    if nan_guard and (not torch.isfinite(dnorm)):
                        raise RuntimeError(format_error_msg("GradNorm", "D", "non-finite gradient norm", rank=rank))
                    optimizer_D.step()

                loss_details["D Adv"] = loss_D.detach().cpu().item()
                if r1_term is not None:
                    loss_details["D R1"] = r1_term.detach().cpu().item()
            
            if use_amp:
                scaler.update()

            scheduler_G.step()
            if gan_enabled and scheduler_D is not None:
                scheduler_D.step()

            train_loss += total_loss.item() * lr.size(0)
            processed_samples += lr.size(0)
            postfix = {k: f"{v:.4f}" for k, v in loss_details.items()}
            postfix["G Main LR"] = f"{optimizer_G.param_groups[0]['lr']:.2e}"
            postfix["G Flow LR"] = f"{optimizer_G.param_groups[1]['lr']:.2e}"
            if gan_enabled and optimizer_D is not None:
                postfix["D LR"] = f"{optimizer_D.param_groups[0]['lr']:.2e}"
            if is_main_process():
                train_pbar.set_postfix(postfix)

                if save_every_iters is not None and global_step % save_every_iters == 0:
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
                        step_in_epoch=step
                    )
    finally:
        for h in handles:
            h.remove()
    
    avg_train_loss = train_loss / processed_samples
    if is_main_process():
        tqdm.write(f"Epoch {epoch+1}/{total_epochs} | Train Loss: {avg_train_loss:.4f}")

    return avg_train_loss, loss_details, global_step


def validate_one_epoch(
    model, ema_model, val_loader, losses, device, config, epoch, lambda_dict
):
    model.eval()
    val_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_imgs = 0
    sample_clean = None
    sample_ema = None
    sample_input = None
    sample_fake = None
    sample_real = None

    total_epochs = config["training"]["epochs"]
    use_ema = config["training"].get("use_ema", False)

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
            want_clean = bool(config["losses"].get("cleaning_charbonnier", False)) or bool(config["losses"].get("cleaning_dists", False))
            out = model(lr, return_lqs=True) if want_clean else model(lr)
            if isinstance(out, (tuple, list)):
                fake, lqs_clean = out
            else:
                fake, lqs_clean = out, None
            fake = fake.view(B * T, C, H, W)
            real = hr.view(B * T, C, H, W)
            if lqs_clean is not None:
                sf = int(config["model"].get("scale_factor", 4))
                gt_clean = F.interpolate(real, scale_factor=1.0 / sf, mode="area", recompute_scale_factor=False)
                pred_clean = lqs_clean.view(B * T, C, H // sf, W // sf).float()

            val_loss_total = 0.0
            if "charbonnier" in losses:
                val_loss_total += lambda_dict["charbonnier"] * losses["charbonnier"](fake, real)
            if "tv" in losses:
                val_loss_total += lambda_dict["tv"] * losses["tv"](fake)
            if config["losses"].get("dists", False):
                fake_dists = torch.clamp((fake + 1.0) * 0.5, 0.0, 1.0).float()
                real_dists = torch.clamp((real + 1.0) * 0.5, 0.0, 1.0).float()
                val_loss_total += lambda_dict["dists"] * losses["dists"](fake_dists, real_dists)
            if config["losses"].get("lpips", False):
                fake_lpips = torch.clamp(fake.float(), -1, 1)
                real_lpips = real.float()
                val_loss_total += lambda_dict["lpips"] * losses["lpips"](fake_lpips, real_lpips)
            if lqs_clean is not None and config["losses"].get("cleaning_charbonnier", False):
                val_loss_total += lambda_dict["cleaning_charbonnier"] * losses["cleaning_charbonnier"](pred_clean, gt_clean)
            if lqs_clean is not None and config["losses"].get("cleaning_dists", False):
                pred01 = torch.clamp((pred_clean + 1.0) * 0.5, 0.0, 1.0)
                gt01   = torch.clamp((gt_clean   + 1.0) * 0.5, 0.0, 1.0)
                val_loss_total += lambda_dict["cleaning_dists"] * losses["cleaning_dists"](pred01, gt01)

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
                    if lqs_clean is not None:
                        sample_clean_lr = lqs_clean[0]
                        sample_clean = F.interpolate(
                            sample_clean_lr.float(),
                            size=(H, W),
                            mode="nearest"
                        )

                    sample_input_lr = lr[0]
                    sample_input = F.interpolate(
                        sample_input_lr.float(),
                        size=(H, W),
                        mode="nearest"
                    )

                    fake_reshaped = fake.view(B, T, C, H, W)
                    sample_fake = fake_reshaped[0]
                    
                    real_reshaped = real.view(B, T, C, H, W)
                    sample_real = real_reshaped[0]

    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_psnr = total_psnr / total_imgs if total_imgs > 0 else 0.0
    avg_ssim = total_ssim / total_imgs if total_imgs > 0 else 0.0
    if is_main_process():
        tqdm.write(f"Epoch {epoch+1}/{total_epochs} | Val Loss: {avg_val_loss:.4f} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")
    
    return avg_val_loss, avg_psnr, avg_ssim, sample_input, sample_fake, sample_real, sample_ema, sample_clean
