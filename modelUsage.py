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
