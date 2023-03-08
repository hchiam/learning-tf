# reference: https://www.tensorflow.org/tutorials/keras/text_classification#sentiment_analysis --> colab

# TODO: extend this code to multi-class text classification and test with data here: https://www.tensorflow.org/tutorials/keras/text_classification#exercise_multi-class_classification_on_stack_overflow_questions
# data, 4 output neurons/classes, tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'], binary_accuracy --> accuracy, val_binary_accuracy --> val_accuracy

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

batch_size = 32
seed = 42  # seed for the randomized training/validation split
max_features = 10000
sequence_length = 250
embedding_dim = 16
epochs = 10

dataset = tf.keras.utils.get_file("aclImdb_v1", "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)
# files or subfolders: ['train', 'test', 'imdbEr.txt', 'README', 'imdb.vocab']

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)
# the train subfolder contains further subfolders:
# ['labeledBow.feat',
#  'pos', <-- contains .txt files of positive reviews, like 1181_9.txt
#  'urls_unsup.txt',
#  'unsup', <-- not needed - delete it below
#  'neg', <-- contains .txt files of negative reviews
#  'urls_neg.txt',
#  'unsupBow.feat',
#  'urls_pos.txt']

# delete the folder /aclImdb/train/unsup:
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
os.listdir(dataset_dir + '/train/') # to confirm that unsup is gone

# split into 80% training data across assumed folder structure (see below):
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',  # aclImdb/train
    batch_size=batch_size, 
    validation_split=0.2,  # 80% training
    subset='training',     # <-- training
    seed=seed)
# assumes this folder structure was already set up before running text_dataset_from_directory:
#       aclImdb/train
#       ...pos
#       ......some_file_1.txt
#       ......some_file_2.txt
#       ...neg
#       ......some_file_1.txt
#       ......some_file_2.txt
# print a few examples for sanity check:
for text_batch, label_batch in raw_train_ds.take(1): # take the 1st batches
  for i in range(3): # print out just the first 3 examples in the batches
    print('Review: ', text_batch.numpy()[i])
    print('Label: ', label_batch.numpy()[i])
print('Label 0 corresponds to: ', raw_train_ds.class_names[0])
print('Label 1 corresponds to: ', raw_train_ds.class_names[1])

# just like we have raw_train_ds, now also create raw validation dataset and raw test dataset:
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2,  # 20% validtion
    subset='validation',   # <-- validation
    seed=seed)
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',  # <-- just use the folder directly
    batch_size=batch_size)

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)  # lower ase
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')  # remove new lines
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')  # remove punctuation

# create a layer to standardize, tokenize, and vectorize input data, to use later (and to include in a separate model later):
vectorize_layer = layers.TextVectorization(  # tokenize and vectorize
    standardize=custom_standardization,  # standardize
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# before actually using that layer, fit it to the training data without labels:
train_text = raw_train_ds.map(lambda x, y: x)  # just text, no labels
vectorize_layer.adapt(train_text)  # adapt = fit text to dictionary. only fit onto training set! don't fit onto test set! otherwise "leaks" info

# now actually use that layer to turn reviews into vectors:
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label  # <-- use vectorize_layer here! to map text to vector of token (using dictionary)
train_ds = raw_train_ds.map(vectorize_text)  # convert dataset text into vectors of tokens!
val_ds = raw_val_ds.map(vectorize_text)      # convert dataset text into vectors of tokens!
test_ds = raw_test_ds.map(vectorize_text)    # convert dataset text into vectors of tokens!

# see an example batch (of 32 reviews and labels) from the dataset:
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print('Review: ', first_review)
print('Label: ', raw_train_ds.class_names[first_label])  # first_label --> class_names as determined by raw_train_ds earlier, see text_dataset_from_directory code above
print('Vectorized review: ', vectorize_text(first_review, first_label))

# see example word tokens converted back into words:
print('token 1287: ',vectorize_layer.get_vocabulary()[1287])  # silent
print('token  313: ',vectorize_layer.get_vocabulary()[313])   # night
print('Vocab size: {}'.format(len(vectorize_layer.get_vocabulary())))  # 10000

# set up data for better performance:
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# set up model:
model = tf.keras.Sequential([                                       # (batch, sequence, embedding), so (None, None, 16) means only explicitly set 1 dim
  layers.Embedding(max_features + 1, embedding_dim),  # output shape: (None, None, 16) and 160016 params
  layers.Dropout(0.2),                                # output shape: (None, None, 16)
  layers.GlobalAveragePooling1D(),                    # output shape: (None, 16)
  layers.Dropout(0.2),                                # output shape: (None, 16)
  layers.Dense(1)])                                   # output shape: (None, 1) and 17 params
model.summary()
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# train model on the training dataset and validation dataset:
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
) # TODO: consider using tf.keras.callbacks.EarlyStopping to detect stable validation accuracy and avoid overfitting on the training data

# evaluate model on the test dataset:
loss, accuracy = model.evaluate(test_ds)
print('Loss: ', loss)
print('Accuracy: ', accuracy)

# plot the history:
history_dict = history.history
history_dict.keys()
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss') # 'bo' = blue dot
plt.plot(epochs, val_loss, 'b', label='Validation loss') # 'b' = solid blue line
plt.title('Training and validation LOSS')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation ACCURACY')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# create another model for export, with the vectorize_layer layer prepended to the model you just trained so it can process raw strings directly:
model_to_export = tf.keras.Sequential([
  vectorize_layer,  # <-- from way earlier!
  model,            # <-- from earlier!
  layers.Activation('sigmoid')
])
model_to_export.compile(
  loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)
# check the accuracy of this new model on raw strings from the raw text test dataset raw_test_ds:
loss, accuracy = model_to_export.evaluate(raw_test_ds)
print(accuracy)
# check that this new model works on raw strings from a "batch" array:
examples = [
  'The movie was great!',
  'The movie was okay.',
  'The movie was terrible...'
]
model_to_export.predict(examples)
