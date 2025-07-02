# SEResNet: Squeeze-and-Excitation ResNet for CIFAR-10 Classification

This repository implements a custom SEResNet (Squeeze-and-Excitation ResNet) architecture in PyTorch for image classification on the CIFAR-10 dataset. The model integrates SE blocks into residual connections to improve representational power by modeling channel-wise relationships.

## Files

- `SEResnet task.ipynb`: Main Jupyter notebook containing full implementation, training, and evaluation.

## Dataset

**CIFAR-10** is a standard dataset of 60,000 32x32 color images across 10 classes (6,000 images per class).  
- 50,000 images for training, 10,000 for testing  
- Automatically downloaded using `torchvision.datasets.CIFAR10`

## Architecture

The model is based on the ResNet structure, with modifications:
- **ResNet Blocks** with identity or downsample skip connections
- **Squeeze-and-Excitation (SE) Blocks**:
  - Global Average Pooling → Fully Connected (bottleneck) → Sigmoid activation → Feature scaling
- Final classifier: Global Average Pooling → Fully Connected layer → Softmax output

## Notebook Workflow

1. **Data Preparation**
   - Load CIFAR-10 dataset using torchvision
   - Normalize images and apply augmentations (random crop, horizontal flip)

2. **Model Construction**
   - Define ResNet and SE blocks from scratch
   - Integrate SE blocks into ResNet layers

3. **Training**
   - Loss: CrossEntropyLoss
   - Optimizer: Adam
   - Epoch-wise accuracy and loss tracking

4. **Evaluation**
   - Accuracy measured on test set
   - Plotting of accuracy/loss curves

## Results

The SEResNet model achieves improved training convergence and better generalization compared to standard ResNet by enhancing feature sensitivity through channel-wise recalibration.

## Author

Adarsh Jupudi  
B.Tech Sophomore, Computer Science and Engineering  
Indian Institute of Technology Bhubaneswar  
LinkedIn: www.linkedin.com/in/adarsh-jupudi

