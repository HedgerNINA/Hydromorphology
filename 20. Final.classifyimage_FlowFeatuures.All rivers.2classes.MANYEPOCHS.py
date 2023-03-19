# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 09:36:34 2022

@author: rdhed
"""

# pip install matplotlib

import matplotlib.pyplot as plt
import numpy as np
#import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib



# DOWNLOAD AND EXPLORE DATASET

#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
#data_dir = pathlib.Path(data_dir)

#data_dir='D:\\DeepLearningDataSets_personal\\Train'
data_dir='E:\\Research\\NINA Projects\\2022_DeepLearning\\TrainingData2class'
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


StandingWaves = list(data_dir.glob('StandingWaves/*'))
PIL.Image.open(str(StandingWaves[0]))

############################################
# LOAD DATA USING A KERAS UTILITY
############################################

# Create a dataset

batch_size = 32
img_height = 100
img_width = 100

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# Visualize the data

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Configure the dataset for performance

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


###############################
# CREATE THE MODEL
###############################

num_classes = len(class_names)
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(num_classes)
])

# Compile the model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Model summary

model.summary()

# Train model

epochs=15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Visualize training results

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')



fig=plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
fig.savefig("E:\\Research\\NINA projects\\2022_DeepLearning\\CNN_Figures\\TraValAccLoss.2class.noaug.MANYEPOCH.png")


## SAVE ACCCURACY AND LOSS


accNP = np.asarray(acc)
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\TrainingResults\\accNP.2class.MANYEPOCH.csv", accNP, delimiter=",",fmt='%s')
val_accNP = np.asarray(val_acc)
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\TrainingResults\\val_accNP.2class.MANYEPOCH.csv", val_accNP, delimiter=",",fmt='%s')
lossNP = np.asarray(loss)
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\TrainingResults\\lossNP.2class.MANYEPOCH.csv", lossNP, delimiter=",",fmt='%s')
val_lossNP = np.asarray(val_loss)
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\TrainingResults\\val_lossNP.2class.MANYEPOCH.csv", val_lossNP, delimiter=",",fmt='%s')


model.save("E:\\Research\\NINA Projects\\2022_DeepLearning\\TrainingResults\\2class_model")
reconstructed_model=keras.models.load_model("E:\\Research\\NINA Projects\\2022_DeepLearning\\TrainingResults\\2class_model")

# Predict on new data


#from PIL import Image
#import os 
#dirname = "D:\\DeepLearningDataSets_personal\\Test\\"
#fname = "178-11.472.jpg"
#fname = "178-01.893.jpg"
#fname = "178-01.26.jpg"



#img = Image.open(os.path.join(dirname, fname))
#img_array = tf.keras.utils.img_to_array(img)
#img_array = tf.expand_dims(img_array, 0) # Create a batch

#predictions = model.predict(img_array)
#score = tf.nn.softmax(predictions[0])
#print(
#    "This image most likely belongs to {} with a {:.2f} percent confidence."
#    .format(class_names[np.argmax(score)], 100 * np.max(score))
#)

# NIDELVA 2019

data_dir='E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Nidelva\\Cells\\LowerNidelva_2019'
data_dir = pathlib.Path(data_dir)
img_height = 100
img_width = 100
Test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  shuffle=False,
  image_size=(img_height, img_width))
count = len(list(data_dir.glob('*/*.jpg')))
PredClass = []
Per = []
predictions = model.predict(Test_ds)
for item in range(count):
   score = tf.nn.softmax(predictions[item])
   PredClass.append ( class_names[np.argmax(score)] )
   Per.append(100 * np.max(score))
FP=Test_ds.file_paths

PredClass = np.asarray(PredClass)
Per = np.asarray(Per)
FP = np.asarray(FP)
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Nidelva\\FlowFeatureResults\\Final.Nidelva_2019.PredClass.2class.MANYEPOCH.csv", PredClass, delimiter=",",fmt='%s')
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Nidelva\\FlowFeatureResults\\Final.Nidelva_2019.FP.2class.MANYEPOCH.csv", FP, delimiter=",",fmt='%s')
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Nidelva\\FlowFeatureResults\\Final.Nidelva_2019.Per.2class.MANYEPOCH.csv", Per, delimiter=",")

# NIDELVA 2020

data_dir='E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Nidelva\\Cells\\LowerNidelva_2020'
data_dir = pathlib.Path(data_dir)
img_height = 100
img_width = 100
Test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  shuffle=False,
  image_size=(img_height, img_width))
count = len(list(data_dir.glob('*/*.jpg')))
PredClass = []
Per = []
predictions = model.predict(Test_ds)
for item in range(count):
   score = tf.nn.softmax(predictions[item])
   PredClass.append ( class_names[np.argmax(score)] )
   Per.append(100 * np.max(score))
FP=Test_ds.file_paths

PredClass = np.asarray(PredClass)
Per = np.asarray(Per)
FP = np.asarray(FP)
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Nidelva\\FlowFeatureResults\\Final.Nidelva_2020.PredClass.2class.MANYEPOCH.csv", PredClass, delimiter=",",fmt='%s')
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Nidelva\\FlowFeatureResults\\Final.Nidelva_2020.FP.2class.MANYEPOCH.csv", FP, delimiter=",",fmt='%s')
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Nidelva\\FlowFeatureResults\\Final.Nidelva_2020.Per.2class.MANYEPOCH.csv", Per, delimiter=",")

# ALTA

data_dir='E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Alta\\Cells'
data_dir = pathlib.Path(data_dir)
img_height = 100
img_width = 100
Test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  shuffle=False,
  image_size=(img_height, img_width))
count = len(list(data_dir.glob('*/*.jpg')))
PredClass = []
Per = []
predictions = model.predict(Test_ds)
for item in range(count):
   score = tf.nn.softmax(predictions[item])
   PredClass.append ( class_names[np.argmax(score)] )
   Per.append(100 * np.max(score))
FP=Test_ds.file_paths

PredClass = np.asarray(PredClass)
Per = np.asarray(Per)
FP = np.asarray(FP)
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Alta\\FlowFeatureResults\Final.Alta.PredClass.2class.MANYEPOCH.csv", PredClass, delimiter=",",fmt='%s')
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Alta\\FlowFeatureResults\\Final.Alta.FP.2class.MANYEPOCH.csv", FP, delimiter=",",fmt='%s')
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Alta\\FlowFeatureResults\\Final.Alta.Per.2class.MANYEPOCH.csv", Per, delimiter=",")

# ORKLA

data_dir='E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Orkla\\Cells'
data_dir = pathlib.Path(data_dir)
img_height = 100
img_width = 100
Test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  shuffle=False,
  image_size=(img_height, img_width))
count = len(list(data_dir.glob('*/*.jpg')))
PredClass = []
Per = []
predictions = model.predict(Test_ds)
for item in range(count):
   score = tf.nn.softmax(predictions[item])
   PredClass.append ( class_names[np.argmax(score)] )
   Per.append(100 * np.max(score))
FP=Test_ds.file_paths

PredClass = np.asarray(PredClass)
Per = np.asarray(Per)
FP = np.asarray(FP)
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Orkla\\FlowFeatureResults\\Final.Orkla.PredClass.2class.MANYEPOCH.csv", PredClass, delimiter=",",fmt='%s')
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Orkla\\FlowFeatureResults\\Final.Orkla.FP.2class.MANYEPOCH.csv", FP, delimiter=",",fmt='%s')
np.savetxt("E:\\Research\\NINA Projects\\2022_DeepLearning\\QGIS\\Orkla\\FlowFeatureResults\\Final.Orkla.Per.2class.MANYEPOCH.csv", Per, delimiter=",")


