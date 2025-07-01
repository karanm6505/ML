# LeNet using PyTorch

A PyTorch implementation of the classic LeNet-5 convolutional neural network, originally designed for handwritten digit recognition, provided in a Jupyter Notebook.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)

## Introduction
LeNet-5 is a pioneering convolutional neural network (CNN) architecture developed by Yann LeCun in 1998. It is widely used for simple image classification tasks, particularly in digit recognition (e.g., MNIST dataset).

This repository provides an implementation of LeNet-5 using PyTorch in a Jupyter Notebook (`LeNet5.ipynb`), including training and evaluation.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/LeNet.git
   cd LeNet
   ```

## Dataset
The implementation uses the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9). The dataset is stored in the `data/MNIST/raw/` directory and is automatically downloaded using `torchvision.datasets` if not already present.

## Usage
### Running the Jupyter Notebook
To use the LeNet-5 model, open the Jupyter Notebook:
```bash
jupyter notebook LeNet5.ipynb
```
Follow the cells in the notebook to train and evaluate the model.

## Training
The notebook contains a training section where you can adjust hyperparameters such as epochs, batch size, and learning rate.

## Evaluation
The trained model can be evaluated using the test dataset within the notebook.

## Results
After training for several epochs, the model typically achieves an accuracy of ~99% on the MNIST dataset.

## References
- Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner, "Gradient-Based Learning Applied to Document Recognition," Proceedings of the IEEE, 1998.
- [LeNet-5 Paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

---




