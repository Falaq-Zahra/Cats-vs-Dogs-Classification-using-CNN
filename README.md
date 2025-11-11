Cats vs Dogs Classification using CNN

Overview
This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs. It covers the full pipeline from data extraction, preprocessing, and exploratory data analysis (EDA) to model training and deployment using Gradio for interactive predictions.The model is trained on a dataset containing separate folders for training and testing images, and it predicts whether an uploaded image is a cat or a dog.

Project Structure
├── data/                   # Extracted dataset
│   ├── train/              # Training images
│   └── test/               # Validation/Test images
├── notebooks/              # Jupyter notebook files
├── README.md               # Project documentation
├── requirements.txt        # Required Python packages
└── model/                  # Optional: Saved trained model

Installation

Clone the repository:
git clone https://github.com/yourusername/cats-vs-dogs-cnn.git
cd cats-vs-dogs-cnn


Install required packages:

pip install -r requirements.txt


The main packages used include:

tensorflow
keras
matplotlib
numpy
gradio

Dataset

The dataset should be organized as follows:
data/
├── train/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/


You can place your dataset as a ZIP file and extract it in the data/ folder.

Usage
1. Data Preparation

Load the images using TensorFlow's image_dataset_from_directory.

Normalize pixel values to be between 0 and 1.

2. Exploratory Data Analysis (EDA)

Check number of images per class.

Visualize class distribution and sample images.

Check image shapes and pixel value ranges.

3. Model Training

Build a 3-layer CNN with convolution, pooling, batch normalization, and dropout.

Compile the model using Adam optimizer and binary_crossentropy loss.

Train the model using the training and validation datasets.

4. Deployment with Gradio

Launch an interactive web app to upload images and get predictions.

The model outputs the predicted class and confidence score.

Example code to launch Gradio interface:

import gradio as gr
import numpy as np
from tensorflow.keras.preprocessing import image

class_names = ['cats', 'dogs']

def predict_image(img):
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    label = class_names[int(pred > 0.5)]
    confidence = pred if label == 'dogs' else 1 - pred
    return {label: float(confidence)}

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Cat or Dog Image"),
    outputs=gr.Label(num_top_classes=2),
    title="Cats vs Dogs Classifier",
    description="Upload an image to see if it's a cat or a dog.",
)
demo.launch()

Model Architecture

Input: 256x256x3 images

Conv Layer 1: 32 filters, 3x3 kernel, ReLU, batch normalization, max pooling

Conv Layer 2: 64 filters, 3x3 kernel, ReLU, batch normalization, max pooling

Conv Layer 3: 128 filters, 3x3 kernel, ReLU, batch normalization, max pooling

Flatten Layer

Dense Layer 1: 128 units, ReLU, Dropout 0.1

Dense Layer 2: 64 units, ReLU, Dropout 0.1

Output Layer: 1 unit, Sigmoid activation (binary classification)

Results

The model achieves reasonable accuracy in classifying cats and dogs.

Performance can be improved with data augmentation, more epochs, or transfer learning.
