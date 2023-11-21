# CNN Digit Reader

## Project Overview
The CNN Digit Reader is a machine learning project designed to recognize handwritten digits. It employs a Convolutional Neural Network (CNN) trained on the MNIST dataset, a collection of 70,000 grayscale images of handwritten digits (0 through 9). The project includes a training module for the CNN model and a graphical user interface (GUI) for uploading and predicting digits from new images.

## Features
- **Digit Recognition Model**: Uses a CNN to learn from the MNIST dataset.
- **GUI for Digit Prediction**: A simple interface to upload images and view predictions.
- **High Accuracy**: Achieves high accuracy on the MNIST validation set.


## Usage
### Training the Model
Run the `training_module.py` script to train the model on the MNIST dataset. This script will download the dataset, train the CNN, and save the model's state for future predictions.

### Using the GUI for Prediction
The GUI is designed to work in a Google Colab environment. Upload the saved model and run the `gui_module.py` script. You can then upload images of handwritten digits to see the model's predictions.

## Project Structure
- `training_module.py`: Contains the CNN model definition, training and validation routines.
- `gui_module.py`: Code for the GUI to upload images and display predictions.
- `digit_cnn_model.pth`: Saved state of the trained CNN model.

## CNN Architecture
The CNN model (`DigitCNN`) includes two convolutional layers followed by max pooling and dropout layers, and two fully connected layers for classification. It uses ReLU activation functions and a softmax output layer for digit prediction.
