# Intel Image Classification Project

A deep learning project that classifies images into six different categories: buildings, forest, glacier, mountain, sea, and street, using the Intel Image Classification dataset from Kaggle.

## Project Overview

This project implements a convolutional neural network (CNN) architecture to classify natural scene images. The model has been trained on the Intel Image Classification dataset and achieves good accuracy in identifying the six different landscape categories.

## Dataset

The project uses the [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dataset from Kaggle, which contains around 25,000 images of size 150x150 distributed under 6 categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

## Model Architecture

The CNN model architecture includes:

- Multiple convolutional layers with batch normalization
- MaxPooling layers
- Dropout layers for regularization
- Dense layers with ReLU activation
- Final softmax layer for 6-class classification

## Features

- Data preprocessing and augmentation
- Model training with early stopping
- Model evaluation and visualization
- Model conversion to different formats:
  - HDF5 (.h5)
  - TensorFlow SavedModel
  - TensorFlow Lite
  - TensorFlow.js

## Project Structure

```
.
├── best_model.h5                # Trained model in HDF5 format
├── requirements.txt             # Python dependencies
├── Submission_Akhir_Pengembangan_Machine_Learning.ipynb # Main notebook
├── model_saved/                 # TensorFlow SavedModel format
│   ├── fingerprint.pb
│   ├── saved_model.pb
│   ├── assets/
│   └── variables/
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── tfjs_model/                  # TensorFlow.js model format
│   ├── group1-shard1of3.bin
│   ├── group1-shard2of3.bin
│   ├── group1-shard3of3.bin
│   └── model.json
└── tflite/                      # TensorFlow Lite model format
    ├── label.txt                # Class labels
    └── model.tflite
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Notebook

The main implementation is in the Jupyter notebook. To run it:

1. Ensure you have Jupyter installed
2. Open the notebook:
   ```bash
   jupyter notebook Submission_Akhir_Pengembangan_Machine_Learning.ipynb
   ```

### Using the Pre-trained Models

The project includes pre-trained models in various formats:

- **HDF5 Model (`best_model.h5`)**: Can be loaded directly with Keras

  ```python
  from tensorflow.keras.models import load_model
  model = load_model('best_model.h5')
  ```

- **TensorFlow Lite (`tflite/model.tflite`)**: For mobile/edge deployment

  ```python
  import tensorflow as tf
  interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
  interpreter.allocate_tensors()
  ```

- **TensorFlow.js (`tfjs_model/`)**: For web browser implementations

## Requirements

The main requirements for this project include:

- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- TensorFlow Lite
- TensorFlow.js

For a full list of dependencies, see `requirements.txt`.

## Author

- Muhammad Thariq Arya Putra Sembiring
- Email: mthariqaryaputra1@gmail.com

## Acknowledgements

- Intel for providing the dataset
- Kaggle for hosting the dataset
- Dicoding Indonesia for the course on Machine Learning Development
