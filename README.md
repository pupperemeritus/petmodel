# GuruNet: A Custom CNN for Cognitive Health Classification

This project introduces GuruNet, a custom-built Convolutional Neural Network (CNN) designed to classify Positron Emission Tomography (PET) scans into five categories of cognitive health. The primary goal is to create a model that is not only highly accurate but also computationally efficient, outperforming standard architectures like ResNet50, InceptionV3, and DenseNet121.

> Feel free to use my model in your projects.

The models classify PET scans into the following five classes:

1. Alzheimer's Disease (AD)
2. Cognitively Normal (CN)
3. Late Mild Cognitive Impairment (LMCI)
4. Early Mild Cognitive Impairment (EMCI)
5. Mild Cognitive Impairment (MCI)

# Models Implemented

This repository contains implementations of the following models:

- GuruNet: A novel, custom-built CNN architecture that combines several modern deep learning concepts for high accuracy and efficiency.
- ResNet50: A pre-trained ResNet50 model, fine-tuned for the PET scan classification task. Uses the built in models from torchvision and modifies the top layer to match the custom model.
- InceptionV3: A pre-trained InceptionV3 model, fine-tuned for the task. Uses the built in models from torchvision and modifies the top layer to match the custom model.
- DenseNet121: A pre-trained DenseNet121 model, fine-tuned for the task. Uses the built in models from torchvision and modifies the top layer to match the custom model.

# File Structure

```
.
├── data/                  # Directory for the dataset
├── checkpoints/            # Saved model checkpoints
├── lightning_logs/        # Logs from TensorBoard
├── dataloader.py          # PyTorch Lightning DataModule for PET scans
├── model.py               # Implementation of the custom GuruNet model
├── other_models.py        # Implementation of ResNet50, InceptionV3, and DenseNet121
├── test_models.py         # Script to test and compare all trained models
└── README.md              # This file
```

# GuruNet Architecture

GuruNet's architecture cherrypicks useful mini-blocks from existing models and slightly modifies them for simplicity and performance:

The backbone uses Inverted Residual Blocks (from MobileNetV2) with integrated Squeeze-and-Excitation (SE) layers to focus on the most informative feature channels. The model incorporates a custom Multi-Scale Block and a depthwise-separable Inception Block to capture both fine-grained details and broader contextual patterns in the PET scans. Dense Blocks (from DenseNet) are used to ensure maximum gradient flow and encourage feature reuse, improving parameter efficiency. Custom Attention Blocks and a Gated Residual Block are placed strategically to help the model focus on the most salient regions of an image and control the flow of information through the network.

# Results

GuruNet not only achieves the highest classification accuracy but does so with significantly fewer parameters and a smaller model size, making it faster and more efficient.

# Model

| Model           | Test Accuracy | Total Params (M) | Model Size (MB) |
| :-------------- | :------------ | :--------------- | :-------------- |
| **GuruNet**     | **98.40%**    | **6.1**          | **24.5**        |
| **DenseNet121** | **97.93%**    | **7.6**          | **30.6**        |
| **ResNet50**    | **97.34%**    | **24.7**         | **98.9**        |
| **InceptionV3** | **96.80%**    | **26.3**         | **105.3**       |

# Prerequisites

Make sure you have Python 3.8+ installed. You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```

Note: A requirements.txt file is not provided, but you can create one with the following contents:

```text
torch
torchvision
pytorch-lightning
pandas
ipython
```

# Installation

Clone the repository:

```bash
git clone https://github.com/pupperemeritus/petmodel
cd petmodel
```

Create a directory for the dataset:

```bash
mkdir data
```

Get the data from [here](https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/pet/)
Place the dataset in the ./data directory, with each class in its own subdirectory (e.g., ./data/AD/, ./data/CN/, etc.).

# Usage

The project is structured using PyTorch Lightning for clean and reproducible training and testing.

# Training Models

You can train each model by running its respective script.

To train GuruNet:

```bash
python model.py
```

To train the other models (e.g., DenseNet121):
Open other_models.py and ensure the desired model training function is called in the if **name** == "**main**": block.

```python
if __name__ == "__main__":

    # Train TORCHVISION_MODEL_NAME
    train_model(TORCHVISION_MODEL_NAME, num_classes=5, batch_size=32, max_epochs=100)

```

Then run the script:

```bash
python other_models.py
```

Training logs will be saved in the lightning_logs/ directory, and the best model checkpoints will be saved in checkpoints/.

Testing Models
To evaluate the performance of all trained models on the test set, run the test_models.py script. This script will automatically find the best-performing checkpoint for each model, run the evaluation, and print the results.

```bash
python test_models.py
```
