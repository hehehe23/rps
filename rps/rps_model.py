from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report ,confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

tf.__version__

dir = '/content/rps'
test_dir = '/content/rps-test-set'

size = 180
batch_size = 32

traindatagen = tf.keras.preprocessing.image_dataset_from_directory(
    dir,
    #validation_split = 0.0,
    image_size=(size, size),
    batch_size = batch_size,
    #subset = 'training',
    color_mode = 'grayscale',
    seed = 321)

#valdatagen = tf.keras.preprocessing.image_dataset_from_directory(
#    dir,
#    validation_split = 0.0,
#    image_size = (size, size),
#    batch_size = batch_size,
#    subset = 'validation',
#    color_mode = 'grayscale',
#    seed = 321)

testdatagen = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size = (size, size),
    batch_size = batch_size,
    color_mode = 'grayscale',
    seed = 321)

class_names = traindatagen.class_names
print(class_names)

model = Sequential()

model.add(Conv2D(32, 3,3, padding = 'same', input_shape = (size ,size ,1)))
model.add(Conv2D(128, 2, activation='relu'))
model.add(Conv2D(64, 2, activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(64, 3,3, padding = 'same', input_shape = (size ,size ,1)))
model.add(Conv2D(64, 2, activation='relu'))
model.add(MaxPool2D(2,2))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))





model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

early_stop = EarlyStopping(patience = 3)

training = model.fit_generator(generator = traindatagen,
                               steps_per_epoch = len(traindatagen),
                               epochs = 10,
                              # validation_data = valdatagen,
                               #validation_steps = len(valdatagen),
                               callbacks = [early_stop])

model.evaluate(testdatagen)

def curve(training, epochs):
  epoch_range = range(1, epochs + 1)
  plt.plot(epoch_range, training.history['accuracy'], color = 'red')
  plt.title('model accurcy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend(['train', 'validation'], loc='lower right',
             bbox_to_anchor=(1.2, 0.3),
             bbox_transform=plt.gcf().transFigure)
  plt.show()

  epoch_range = range(1, epochs + 1)
  plt.plot(epoch_range, training.history['loss'], color = 'red')
  plt.title('model loss')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(['train', 'validation'], loc='lower right',
             bbox_to_anchor=(1.2, 0.3),
             bbox_transform=plt.gcf().transFigure)
  plt.show()

  curve(training, 10)
