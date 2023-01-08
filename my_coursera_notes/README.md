# My notes for a Coursera course

- course: https://www.coursera.org/learn/introduction-tensorflow

- slides: https://community.deeplearning.ai/t/tf1-course-1-lecture-notes/124222

- **DNN** = Deep Neural Network
- **ConvNet** = Convolutional neural Network

- **convolution** = filtering image to emphasize useful features like vertical lines (example of convolution: [`keras.layers.Conv2D`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D))

- **pooling** = compressing image data while detecting regions with features detected by convolution (example of pooling: [`keras.layers.MaxPool2D`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D))

## simple mapping of x -> y with a single layer with one neuron

```py
import tensorflow as tf
import numpy as np
from tensorflow import keras

xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xs, ys, epochs=1000)

new_x = 7.0
prediction = model.predict([new_x])[0]
print(prediction)
```

## naive image recognition of Fashion MNIST

```py
import os
import tensorflow as tf
from tensorflow import keras

class myCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    # if logs.get('loss') < 0.4:
    if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
      self.model.stop_training = True
      print('\nStopped training - good enough.')

callbacks = myCallback()

mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# # or if the data's already downloaded:
# current_dir = os.getcwd()
# data_path = os.path.join(current_dir, "data/mnist.npz")
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path=data_path)
train_images = train_images / 255.0
test_images  = test_images  / 255.0

data_shape = train_images.shape
print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")

model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28,28)), # 28*28 = 784 input neurons
  keras.layers.Dense(512, activation=tf.nn.relu), # 512 neurons with ReLU to learn non-linear relationships quickly
  keras.layers.Dense(10,  activation=tf.nn.softmax), # 10 neurons with softmax to smoothen to probabilities for classification
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metric=['accuracy'])
model.fit(train_images, train_labels, epochs=5, callbacks=callbacks) # run 5x max, unless hit threshold loss on_epoch_end
```

## image recognition of Fashion MNIST using convolution and pooling

```py
# ...
model = keras.Sequential([
  # detect features:
  keras.layers.Conv2D(64, (3,3), activation='relu', # 64 = 64 filters
                      input_shape=(28,28,1)),
  keras.layers.MaxPool2D(2,2),
  # detect higher-level features:
  keras.layers.Conv2D(64, (3,3), activation='relu'), # 64 = 64 filters
  keras.layers.MaxPool2D(2,2),

  # similar DNN layers as before:
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation=tf.nn.relu),
  keras.layers.Dense(10,  activation=tf.nn.softmax),
])

model.summary() # prints out layer types and shapes/sizes
# note that convolution layer shapes will be smaller because filters can't reach outside of the bounds of the image
# ...
```

```py
def reshape_and_normalize(images):
  # images array is already 3D because it's a list of images, but
  # add "1" for extra dimension for RGB:
  images = np.reshape(images, (len(images), len(images[0]), len(images[0][0]), 1))

  # normalize to 0-1 range:
  images = images / 255.0

  return images
```
