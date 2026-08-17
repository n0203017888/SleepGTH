"""Supervised training loop + evaluator."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score



def train_one_epoch(model, criterion, loader: DataLoader, optimizer, device,
                       scheduler=None, scaler=None,
                       log_every: int = 50,
                       max_steps: int | None = None) -> dict:
    model.train()
    losses = []
    total = max_steps if max_steps is not None else len(loader)
    pbar = tqdm(loader, desc='train', total=total)
    for step, batch in enumerate(pbar):
        if max_steps is not None and step >= max_steps:
            break
        batch = {k: v.to(device, non_blocking=True)
                 for k, v in batch.items() if torch.is_tensor(v)}
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast(device.type, dtype=torch.float16):
                out = model(batch)
                loss = criterion(out['epoch_logits'], batch['label'])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(batch)
            loss = criterion(out['epoch_logits'], batch['label'])
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
        if (step + 1) % log_every == 0:
            pbar.set_postfix(
                loss=f'{sum(losses[-log_every:]) / log_every:.4f}',
                lr=f'{optimizer.param_groups[0]["lr"]:.2e}',
            )
    return {'loss_mean': sum(losses) / len(losses)}


def _metric_dict(true, pred) -> dict:
    return {
        'accuracy': accuracy_score(true, pred),
        'kappa': cohen_kappa_score(true, pred),
        'macro_f1': f1_score(true, pred, average='macro', zero_division=0),
        'per_class_f1': f1_score(true, pred, average=None, zero_division=0).tolist(),
        'confusion_matrix': confusion_matrix(true, pred, labels=[0, 1, 2, 3, 4]).tolist(),
        'n_samples': len(true),
    }


@torch.no_grad()
def evaluate(model, loader: DataLoader, device,
             max_steps: int | None = None, progress: bool = False) -> dict:
    model.eval()
    all_logits = []
    all_labels = []
    iterator = tqdm(loader, desc='eval', total=max_steps or len(loader)) if progress else loader
    for step, batch in enumerate(iterator):
        if max_steps is not None and step >= max_steps:
            break
        batch = {k: v.to(device, non_blocking=True)
                 for k, v in batch.items() if torch.is_tensor(v)}
        out = model(batch)
        all_logits.append(out['epoch_logits'].cpu().numpy())
        all_labels.append(batch['label'].cpu().numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    raw_pred = logits.argmax(axis=-1)
    return _metric_dict(labels, raw_pred)
