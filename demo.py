import torch
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import time
import numpy as np
import matplotlib.pyplot as plt

from models.model import SimpleSOD
from config import DEVICE

# Load model
model = SimpleSOD().to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict(image):

    start_time = time.time()

    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img)

    pred = pred.squeeze().cpu().numpy()

    # binary mask
    mask = (pred > 0.5).astype(np.uint8)

    # resize for display
    image = image.resize((128, 128))

    # overlay
    overlay = np.array(image) * 0.6 + np.stack([mask*255]*3, axis=-1) * 0.4
    overlay = overlay.astype(np.uint8)

    inference_time = time.time() - start_time

    return mask * 255, overlay, f"{inference_time:.4f} sec"


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(label="Predicted Mask"),
        gr.Image(label="Overlay"),
        gr.Text(label="Inference Time")
    ],
    title="Salient Object Detection Demo"
)

interface.launch()