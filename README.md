# CNN Digit Reader

## Overview
The CNN Digit Reader is an advanced machine learning project focused on recognizing handwritten digits. It uses a Convolutional Neural Network (CNN), trained on the MNIST dataset, to classify digits from 0 to 9. This project demonstrates the application of deep learning techniques in computer vision.

## Key Features
- **Digit Recognition CNN Model**: Utilizes a CNN for high-accuracy digit recognition.
- **MNIST Dataset Training**: Trained on the well-known MNIST dataset of handwritten digits.
- **Interactive GUI**: Offers a user-friendly interface for digit prediction in Google Colab.

## Technical Stack
- **Python**: The primary programming language used.
- **PyTorch**: A deep learning framework for model construction and training.
- **OpenCV (cv2)**: For image processing and manipulation in the training module.
- **PIL (Pillow)**: Used in the GUI module for image processing.
- **NumPy**: For numerical operations, especially in image transformations.
- **Google Colab**: Cloud-based platform used for GUI deployment and interaction.


## Training Module
`training_module.py` includes the CNN architecture and training logic.
- **CNN Architecture**: The model comprises convolutional layers, max-pooling layers, dropout layers, and fully connected layers. It uses ReLU activation and a softmax output layer.
- **Data Preprocessing**: Images are resized and normalized. OpenCV is used for initial image manipulation.
- **Training Process**: Employs the Adam optimizer, learning rate scheduling, and backpropagation.
- **Regularization**: Dropout layers are included to prevent overfitting.

## GUI Module
`gui_module.py` is designed for Google Colab and provides a simple interface for uploading images and viewing predictions.
- **Image Upload and Processing**: Users can upload images, which are then processed using PIL for compatibility with the CNN model.
- **Prediction**: The pre-trained model predicts the digit and displays the result.

## Usage
1. **Train the Model**: Run `training_module.py` to train the model.
2. **Predict Using GUI**: Upload the trained model to Google Colab and run `gui_module.py` to start making predictions.
3. **NOTE**: Ensure the pictures provided adhere to the MNIST dataset. You can train this model on other datasets. 
