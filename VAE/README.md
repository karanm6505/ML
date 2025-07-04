# Variational Autoencoder (VAE) for MNIST Image Generation

This project implements a Variational Autoencoder (VAE) in Python using PyTorch. The model is trained on the MNIST dataset of handwritten digits and can be used to reconstruct existing digits and generate new ones.

## Project Structure

```
VAE/
├── model.py            # Defines the VAE architecture
├── train.py            # Script for training the VAE model
├── dataset/            # Contains the MNIST dataset
│   └── MNIST/
│       ├── raw/        # Raw MNIST data files
│       └── ...
├── generated_images/   # Stores images generated by the trained VAE
│   ├── generated_0_ex0.png
│   └── ...
└── README.md           # This file
```

## Setup and Installation

1.  **Prerequisites**:
    *   Python (tested with 3.13)
    *   pip (Python package installer)

2.  **Create a Virtual Environment** (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `\.venv\\Scripts\\activate`
    ```

3.  **Install Dependencies**:
    The primary dependency is PyTorch. You may also need `torchvision` for dataset handling.
    ```bash
    pip install torch torchvision
    ```
    If you encounter issues with PyTorch installation, please refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for instructions specific to your system.

## Usage

1.  **Training the Model**:
    To train the VAE, run the `train.py` script:
    ```bash
    python train.py
    ```
    This script will typically load the MNIST dataset, initialize the VAE model defined in `model.py`, train it, and save the trained model weights (you might need to add this functionality to `train.py` if not already present).

2.  **Generating Images**:
    After training, you can use the model (potentially through `model.py` or a separate inference script) to generate new images. The `generated_images/` folder is intended to store such outputs. The `model.py` script currently demonstrates a forward pass with random data.

## Model Architecture

The VAE is defined in `model.py` and consists of:
*   An **Encoder**: Maps input images to a latent space distribution (mean and log-variance).
*   A **Reparameterization Trick**: Samples from the latent distribution.
*   A **Decoder**: Maps samples from the latent space back to image space.

## Output

Generated images, such as samples from the latent space or reconstructions of input data, can be found in the `generated_images/` directory.

---
