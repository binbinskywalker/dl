# from tensorflow.keras.datasets import mnist
import tensorflow  as tf
import keras

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
# Test basic functionality
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(1,))
])
model.compile(optimizer='adam', loss='mean_squared_error')

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images.shape