# Handwritten Digits Recognition

This project aims to recognize handwritten digits using machine learning techniques. It utilizes the MNIST dataset and a deep neural network model implemented with TensorFlow and Keras.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Digit Recognition](#digit-recognition)
- [License](#license)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Maharnab-Saikia/HandwrittenDigitsAI.git
cd HandwrittenDigitsAI
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To use the project, follow the steps below:

1. Ensure that the required dependencies are installed (see [Installation](#installation)).

2. Run the following command to perform digit recognition using the provided images in the `digits/` folder:

```bash
python handwritten_digits_recognition.py
```

This command will load the trained model (if available) or train a new model on the MNIST dataset. It will then perform digit recognition on the images in the `digits/` folder, displaying the predicted digit along with the corresponding image.

## Model Training

The model is trained on the MNIST dataset, which consists of 60,000 training samples and 10,000 test samples of handwritten digits.

If the `model` directory exists, the pre-trained model will be loaded. Otherwise, a new model will be created and trained using the training data.

The model architecture consists of a flattened input layer, two dense hidden layers with ReLU activation, and a dense output layer with softmax activation for the 10 digits.

## Digit Recognition

The script performs digit recognition on the images in the `digits/` folder. Each image is read using OpenCV, preprocessed, and fed into the trained model for prediction. The predicted digit is displayed along with the corresponding image.

Feel free to add your own images to the `digits/` folder for digit recognition.

## License

This project is licensed under the [MIT License](LICENSE).