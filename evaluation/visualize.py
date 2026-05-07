import os
import matplotlib.pyplot as plt

def save_prediction(img, mask, pred, idx, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    img = img.permute(1, 2, 0).cpu()
    mask = mask.squeeze().cpu()
    pred = pred.squeeze().cpu()

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))

    ax[0].imshow(img)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("GT")
    ax[1].axis("off")

    ax[2].imshow(pred, cmap="gray")
    ax[2].set_title("Pred")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/result_{idx}.png")
    plt.close()