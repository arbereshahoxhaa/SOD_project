import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3

TRAIN_IMG_DIR = "data/images/train"
TRAIN_MASK_DIR = "data/masks/train"

TEST_IMG_DIR = "data/images/test"
TEST_MASK_DIR = "data/masks/test"

SAVE_MODEL_PATH = "best_model.pth"
PRED_SAVE_DIR = "outputs/predictions"