import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(device_arg: str = "auto"):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)

def train_one_epoch(model, loader, optimizer, device, epoch: int, desc: str = "train"):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"{desc} epoch {epoch}", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
        pbar.set_postfix(loss=total_loss / total, acc=100.0 * total_correct / total)

    return total_loss / total, 100.0 * total_correct / total

@torch.no_grad()
def evaluate(model, loader, device, desc: str = "eval"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, 100.0 * total_correct / total
