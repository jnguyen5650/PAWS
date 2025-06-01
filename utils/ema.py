import torch

def update_ema(ema_model, model, alpha=0.99):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data = alpha * ema_param.data + (1 - alpha) * param.data