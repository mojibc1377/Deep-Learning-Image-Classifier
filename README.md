
# Image Classification with Deep Learning: CIFAR-10 Dataset

## Introduction
This project aims to build a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal is to classify these images into their respective categories.

## Dataset
The CIFAR-10 dataset is a popular benchmark dataset in machine learning. It consists of 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Data Preprocessing
The data was preprocessed by normalizing the pixel values to a range of 0 to 1. This helps in speeding up the training process and achieving better convergence.

```python
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images to a range of 0 to 1
x_train, x_test = x_train / 255.0, x_test / 255.0
```

## Model Architecture
The CNN architecture used in this project is as follows:
- Conv2D layer with 32 filters, kernel size 3x3, ReLU activation
- MaxPooling2D layer with pool size 2x2
- Conv2D layer with 64 filters, kernel size 3x3, ReLU activation
- MaxPooling2D layer with pool size 2x2
- Conv2D layer with 64 filters, kernel size 3x3, ReLU activation
- Flatten layer
- Dense layer with 64 units, ReLU activation
- Dropout layer with 0.5 dropout rate
- Dense layer with 10 units, softmax activation

The model was compiled using the Adam optimizer and sparse categorical cross-entropy loss.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
```

## Model Training
The model was trained for 10 epochs, using the training data. The training and validation accuracy and loss were recorded at each epoch.

```python
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))
```

## Model Evaluation
The model was evaluated on the test data, achieving a test accuracy of approximately 70%.

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")
```

## Visualization

### Model Accuracy
This graph shows the training and validation accuracy over the epochs. The model's accuracy improves steadily, with the validation accuracy closely following the training accuracy.

<img width="1200" alt="Screenshot 2024-05-25 at 16 21 52" src="https://github.com/mojibc1377/ML-model-train-evaluation-for-Image-classification/assets/82224660/c368cfdd-4dbf-4455-8cb9-19c2fd2aedd4">


**Explanation**:
- The x-axis represents the number of epochs.
- The y-axis represents the accuracy.
- The blue line shows the training accuracy, while the orange line shows the validation accuracy.
- The accuracy increases as the number of epochs increases, indicating that the model is learning and improving its performance.

### Model Loss
This graph shows the training and validation loss over the epochs. The model's loss decreases steadily, indicating that the model is learning and improving over time.

<img width="1200" alt="Screenshot 2024-05-25 at 16 22 01" src="https://github.com/mojibc1377/ML-model-train-evaluation-for-Image-classification/assets/82224660/3878d68d-0bbd-4472-be6c-c407eef72d17">


**Explanation**:
- The x-axis represents the number of epochs.
- The y-axis represents the loss.
- The blue line shows the training loss, while the orange line shows the validation loss.
- The loss decreases as the number of epochs increases, indicating that the model is reducing its prediction error and improving its performance.

## Conclusion
The CNN model achieved a test accuracy of approximately 70% on the CIFAR-10 dataset. This indicates that the model can reasonably classify images into the correct categories, although there is room for improvement. Potential improvements could include fine-tuning the hyperparameters, using data augmentation, and experimenting with different architectures.

## Model Usage
To use the trained model for classifying new images, follow these steps:

```python
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model('cifar10_cnn_model.h5')

def preprocess_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(32, 32))
    # Convert the image to an array
    img_array = image.img_to_array(img)
    # Scale pixel values to the range 0-1
    img_array /= 255.0
    # Expand dimensions to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Example usage
img_path = input("path_to_your_image.jpg:")
img_array = preprocess_image(img_path)

# Make a prediction
predictions = model.predict(img_array)
# Get the class with the highest probability
predicted_class = np.argmax(predictions, axis=1)

# Map the predicted class to the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(f'Predicted class: {class_names[predicted_class[0]]}')
```

## How to Run
1. Clone this repository.
2. Install the necessary libraries: `pip install tensorflow numpy matplotlib`
3. Run the Jupyter Notebook: `jupyter notebook image_classification_cifar10.ipynb`

This project demonstrates the process of building, training, and evaluating a CNN for image classification on a well-known benchmark dataset.
