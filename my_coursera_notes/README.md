# My notes for DeepLearning.AI / Coursera course - intro to TF for AI/ML/DL

- course: https://www.coursera.org/learn/introduction-tensorflow

- slides: https://community.deeplearning.ai/t/tf1-course-1-lecture-notes/124222

- companion repo: https://github.com/https-deeplearning-ai/tensorflow-1-public/tree/main/C1 - Jupyter notebooks not rendering for you on GitHub? Try https://nbviewer.org

- discussion forum: https://community.deeplearning.ai/c/tf1/tf1-course-2/80

- my own notes from another crash course:
  https://github.com/hchiam/machineLearning/blob/master/more_notes/googleMLCrashCourse.md

- **DNN** = Deep Neural Network
- **CNN** or **ConvNet** = Convolutional neural Network

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

## less-naive image recognition of Fashion MNIST using convolution and pooling

```py
# ...
model = keras.models.Sequential([
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

## more realistic binary classification image recognition using Keras ImageDataGenerator to generate images

- realistic images don't have perfectly-same-scaled views of objects filling their images in the center
- make sure to make `flow_from_directory` point to `/training` and not to `/training/label-name-1`
  - folders: `/training`
  - folders: `/training/label-name-1`
  - folders: `/training/label-name-2`
  - folders: `/validation`
  - folders: `/validation/label-name-1`
  - folders: `/validation/label-name-2`

```py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
  train_dir, # /training folder, not /training/label-name-1
  target_size=(300,300), # will need to match this number later
  batch_size=128, # will need this number later
  class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
  validation_dir, # /validation folder, not /validation/label-name-1
  target_size=(300,300), # will need to match this number later
  batch_size=32, # will need this number later
  class_mode='binary'
)
```

```py
# ...
model = keras.models.Sequential([
  # detect features:
  keras.layers.Conv2D(16, (3,3), activation='relu', # 16 = 16 filters
                      input_shape=(300,300,3)), # 3 for RGB, 300x300 from generator
  keras.layers.MaxPool2D(2,2),
  # detect higher-level features:
  keras.layers.Conv2D(32, (3,3), activation='relu'), # 32 = 32 filters
  keras.layers.MaxPool2D(2,2),
  # detect higher-higher-level features:
  keras.layers.Conv2D(64, (3,3), activation='relu'), # 64 = 64 filters
  keras.layers.MaxPool2D(2,2),

  # similar DNN layers as before:
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation=tf.nn.relu),
  keras.layers.Dense(1,  activation=tf.nn.sigmoid), # if doing binary classification
])

model.summary() # prints out layer types and shapes/sizes
# note that convolution layer shapes will be smaller because filters can't reach outside of the bounds of the image
# ...
```

```py
from tensorflow.keras.optimizers import RMSprop # lets you play with learning rate

# ...

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metric=['accuracy'])

history = model.fit(
  train_generator, # (see generator code snippet above)
  steps_per_epoch=8, # 8 = 1028 images in training folder / 128 batch_size from train_generator
  epochs=15,
  validation_data=validation_generator, # (see generator code snippet above)
  validation_steps=8, # 8 = 256 images in validation folder / 32 batch_size from validation_generator
  verbose=2
)
```

## Next steps

[C2](https://github.com/hchiam/learning-tensorflow/blob/main/my_coursera_notes/C2.md).md: https://www.coursera.org/learn/convolutional-neural-networks-tensorflow

[C3](https://github.com/hchiam/learning-tensorflow/blob/main/my_coursera_notes/C3.md).md: https://www.coursera.org/learn/natural-language-processing-tensorflow

[C4](https://github.com/hchiam/learning-tensorflow/blob/main/my_coursera_notes/C4.md).md: https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction

https://github.com/hchiam/learning-pycharm

For the TF cert exam:

- https://www.tensorflow.org/certificate
- https://www.tensorflow.org/static/extras/cert/TF_Certificate_Candidate_Handbook.pdf
- https://www.tensorflow.org/extras/cert/Setting_Up_TF_Developer_Certificate_Exam.pdf
- https://www.jetbrains.com/help/pycharm/installation-guide.html
- https://plugins.jetbrains.com/plugin/13812-tensorflow-developer-certificate/versions
