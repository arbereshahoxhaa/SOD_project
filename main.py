import torch
from torch.utils.data import DataLoader
import numpy as np

from config import *
from dataset.dataset import SODDataset
from models.model import SimpleSOD
from training.train import train_model
from evaluation.visualize import save_prediction
from evaluation.metrics import iou_score
from evaluation.experiments import log_result, save_results

import torch.nn as nn


# -------------------------
# DATA
# -------------------------
train_dataset = SODDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, IMAGE_SIZE)
test_dataset = SODDataset(TEST_IMG_DIR, TEST_MASK_DIR, IMAGE_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# -------------------------
# MODEL
# -------------------------
device = DEVICE
model = SimpleSOD().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# -------------------------
# TRAINING LOOP
# -------------------------
for epoch in range(EPOCHS):

    loss = train_model(model, train_loader, optimizer, device)

    print(f"\nEpoch {epoch+1} | Loss: {loss:.4f}")


    # -------------------------
    # EVALUATION
    # -------------------------
    model.eval()

    iou_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    with torch.no_grad():

        for idx, (img, mask) in enumerate(test_loader):

            img = img.to(device)
            mask = mask.to(device)

            pred = model(img)
            pred_bin = (pred > 0.5).float()


            # -------------------------
            # IoU
            # -------------------------
            iou = iou_score(pred_bin, mask).item()
            iou_list.append(iou)


            # -------------------------
            # Precision / Recall / F1
            # -------------------------
            tp = (pred_bin * mask).sum()
            fp = (pred_bin * (1 - mask)).sum()
            fn = ((1 - pred_bin) * mask).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = (2 * precision * recall) / (precision + recall + 1e-8)

            precision_list.append(precision.item())
            recall_list.append(recall.item())
            f1_list.append(f1.item())


            # -------------------------
            # SAVE VISUALIZATION
            # -------------------------
            save_prediction(
                img[0],
                mask[0],
                pred_bin[0],
                idx,
                "outputs/predictions"
            )


    # -------------------------
    # AVERAGES
    # -------------------------
    mean_iou = np.mean(iou_list)
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    mean_f1 = np.mean(f1_list)


    print(f"Val IoU: {mean_iou:.4f}")
    print(f"Precision: {mean_precision:.4f}")
    print(f"Recall: {mean_recall:.4f}")
    print(f"F1-score: {mean_f1:.4f}")


    # -------------------------
    # SAVE MODEL
    # -------------------------
    torch.save(model.state_dict(), "best_model.pth")
    print("Model saved!")


    # -------------------------
    # EXPERIMENT LOGGING
    # -------------------------
    log_result(
        name=f"Epoch {epoch+1} CNN",
        iou=mean_iou,
        f1=mean_f1
    )


# -------------------------
# SAVE EXPERIMENT RESULTS
# -------------------------
save_results()

print("\nTraining Complete!")