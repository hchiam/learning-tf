# extra notes

if

```py
def preprocess(image, label): # to resize & normalize images
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  # image = np.reshape(image, (IMG_SIZE, IMG_SIZE, 3))
  image = image / .255
  return image, label

training_dataset = training_dataset
  .map(preprocess) # to resize & normalize images
  .cache()
  .shuffle(dataset_info.splits['train'].num_examples)
  .batch(BATCH_SIZE)
  .prefetch(tf.data.experimental.AUTOTUNE)

validation_dataset = validation_dataset
  .map(preprocess) # to resize & normalize images
  .batch(BATCH_SIZE)
  .cache()
  .prefetch(tf.data.experimental.AUTOTUNE)
```

then

```py
model.fit(training_dataset, epochs=15, verbose=2)
```
