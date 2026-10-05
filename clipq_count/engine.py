"""Training / validation / test loops (AMP, grad clipping, cosine LR, early stopping)."""
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .loss import CombinedCountingLoss
from .utils import clear_memory, logger, move_batch


def _forward(model, criterion, batch, device):
    with autocast(device_type=device.type):
        count, log_count, density = model(batch["images"], batch["examples"], batch["text"], batch["points"])
        loss = criterion(count, batch["counts"], log_count.squeeze(-1), batch["log_counts"], density, batch["density"])
    return count, loss


@torch.no_grad()
def run_eval(model, loader, device, criterion=None, desc="Validation"):
    """Return (mean MAE, precision within +/-1, mean loss, predictions, targets)."""
    model.eval()
    criterion = criterion or CombinedCountingLoss()
    mae = loss_sum = 0.0
    correct = total = 0
    preds, targets = [], []
    for raw in tqdm(loader, desc=desc):
        batch = move_batch(raw, device)
        count, loss = _forward(model, criterion, batch, device)
        err = torch.abs(count - batch["counts"])
        mae += err.mean().item()
        correct += (err <= 1).sum().item()
        total += batch["counts"].numel()
        loss_sum += loss.item()
        preds.extend(count.float().cpu().numpy())
        targets.extend(batch["counts"].cpu().numpy())
    n = max(len(loader), 1)
    return mae / n, (correct / total if total else 0.0), loss_sum / n, np.array(preds), np.array(targets)


def train(model, train_loader, val_loader, device, *, epochs, lr, weight_decay, patience, checkpoint_path, log_every=20):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    logger.info("Training %d parameters", sum(p.numel() for p in params))

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    criterion = CombinedCountingLoss()

    history = {"train_mae": [], "val_mae": [], "train_prec": [], "val_prec": []}
    best_val_mae, bad_epochs = float("inf"), 0

    for epoch in range(1, epochs + 1):
        model.train()
        mae = loss_sum = 0.0
        correct = total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, raw in enumerate(pbar):
            batch = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            count, loss = _forward(model, criterion, batch, device)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            err = torch.abs(count - batch["counts"])
            mae += err.mean().item()
            correct += (err <= 1).sum().item()
            total += batch["counts"].numel()
            loss_sum += loss.item()
            if step % log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{err.mean().item():.2f}",
                                 prec=f"{correct / max(total, 1):.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                clear_memory()

        n = max(len(train_loader), 1)
        train_mae, train_prec = mae / n, correct / max(total, 1)
        val_mae, val_prec, val_loss, _, _ = run_eval(model, val_loader, device, criterion)
        for k, v in zip(history, (train_mae, val_mae, train_prec, val_prec)):
            history[k].append(v)
        logger.info("Epoch %d: train loss %.4f mae %.2f prec %.3f | val loss %.4f mae %.2f prec %.3f",
                    epoch, loss_sum / n, train_mae, train_prec, val_loss, val_mae, val_prec)

        if val_mae < best_val_mae:
            best_val_mae, bad_epochs = val_mae, 0
            torch.save({"model_state_dict": model.state_dict(), "best_val_mae": best_val_mae, "epoch": epoch}, checkpoint_path)
            logger.info("Saved best checkpoint (val MAE %.2f) -> %s", best_val_mae, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("Early stopping after %d epochs without improvement", patience)
                break
        scheduler.step()
        clear_memory()
    return history
