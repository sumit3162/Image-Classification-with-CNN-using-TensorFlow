# Image Classification with CNN using TensorFlow

## Overview

This project implements a Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras. The model is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Features

- **CNN Architecture**: Implements a 3-layer convolutional neural network with max pooling
- **Data Preprocessing**: Includes automatic normalization of pixel values (0-255 → 0-1)
- **Model Training**: Uses Adam optimizer with sparse categorical crossentropy loss
- **Evaluation**: Provides test accuracy metrics after training

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- CIFAR-10 dataset (automatically downloaded)

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd image-classification-cnn
   ```

2. Install the required packages:
   ```bash
   pip install tensorflow numpy
   ```

## Usage

1. Run the Jupyter notebook or Python script:
   ```bash
   jupyter notebook imcvnn.ipynb
   ```
   or
   ```bash
   python imcvnn.py
   ```

2. The script will:
   - Automatically download the CIFAR-10 dataset
   - Preprocess the images
   - Train the CNN model
   - Evaluate on test data
   - Output the test accuracy

## Model Architecture

The CNN consists of:
1. Conv2D layer (32 filters, 3x3 kernel, ReLU activation)
2. MaxPooling2D layer (2x2 pool size)
3. Conv2D layer (64 filters, 3x3 kernel, ReLU activation)
4. MaxPooling2D layer (2x2 pool size)
5. Conv2D layer (64 filters, 3x3 kernel, ReLU activation)
6. Flatten layer
7. Dense layer (64 units, ReLU activation)
8. Output layer (10 units for 10 classes)

## Training Parameters

- Epochs: 10
- Batch size: Default (32)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Validation split: 20% of training data

## Expected Output

After training, you should see output similar to:
```
Test accuracy: 0.7012
```

## Customization Options

- Adjust the number of epochs in the `model.fit()` call
- Modify the CNN architecture by adding/removing layers
- Change the optimizer or learning rate
- Try different activation functions
- Add regularization (Dropout, BatchNorm) to prevent overfitting

## Future Improvements

- Implement data augmentation to improve generalization
- Add model checkpointing and early stopping
- Experiment with different architectures (ResNet, VGG, etc.)
- Add visualization of training progress and sample predictions
- Extend to other datasets beyond CIFAR-10

## License

This project is open-source and available under the MIT License.

