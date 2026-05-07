import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, loader, optimizer, device):

    model.train()

    bce = nn.BCELoss()

    total_loss = 0

    for images, masks in tqdm(loader):

        images = images.to(device)
        masks = masks.to(device)

        # 🔥 THIS WAS MISSING IN YOUR CODE
        preds = model(images)

        # loss
        bce_loss = bce(preds, masks)

        # dice loss
        dice_loss = 1 - (2 * (preds * masks).sum() + 1e-8) / ((preds + masks).sum() + 1e-8)

        loss = bce_loss + dice_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)