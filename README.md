# Salient Object Detection (SOD) Project

## Project Overview
This project implements a Deep Learning-based Salient Object Detection (SOD) system using Convolutional Neural Networks (CNNs). The goal is to perform pixel-level segmentation of images and generate binary masks that highlight the most important objects in a scene.

The model is trained to distinguish foreground (salient objects) from background using supervised learning on image-mask pairs.

---

## Objectives
- Build an end-to-end deep learning pipeline for image segmentation
- Train a CNN model for pixel-wise prediction
- Generate segmentation masks for input images
- Evaluate performance using IoU and F1-score metrics

---

## Technologies Used
- Python
- PyTorch
- NumPy
- OpenCV
- Matplotlib
- Google Colab / GPU training

---

## Project Structure
SOD_Project/
│
├── data/ # Dataset (NOT included in GitHub)
├── models/ # Model architecture
├── train.py # Training script
├── demo.py # Testing / inference
├── utils.py # Helper functions
├── requirements.txt # Dependencies
└── README.md

## Model Architecture
- CNN-based encoder network
- Convolution layers with ReLU activation
- Sigmoid output for pixel-wise probability
- Binary segmentation (foreground vs background)

---

## Training Details
- Loss Function: Binary Cross Entropy (BCE)
- Optimizer: Adam
- Dataset Split: 70% Train / 15% Validation / 15% Test
- Training performed using GPU acceleration

---

## Evaluation Metrics
- Intersection over Union (IoU)
- F1 Score

### Results:
- Mean IoU: ~0.11  
- Mean F1 Score: ~0.17  

---

## Limitations
- Limited dataset size
- Simple CNN architecture
- Struggles with complex backgrounds and fine object boundaries

---

## Future Improvements
- Use advanced architectures like U-Net
- Apply data augmentation techniques
- Train on larger datasets
- Improve loss functions (Dice + IoU combined)

---

## Dataset
The dataset is not included in this repository due to size limitations.

To run the project:
1. Download or prepare dataset
2. Place it inside the `data/` folder
3. Ensure image-mask structure is maintained

---

## How to Run
```bash
pip install -r requirements.txt
python train.py
python demo.py