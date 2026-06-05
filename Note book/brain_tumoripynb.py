import zipfile
zip = zipfile.ZipFile('/content/archive (3).zip')
zip.extractall('/content/')
zip.close()

from tensorflow.keras.utils import image_dataset_from_directory

train_dataset = image_dataset_from_directory(
    "/content/Training",
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

valid_dataset = image_dataset_from_directory(
    "/content/Testing",
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

train_dataset.class_names

import tensorflow as tf

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
valid_dataset = valid_dataset.map(lambda x, y: (normalization_layer(x), y))

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[-30:]:
    layer.trainable = True

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3
)

history = model.fit(train_dataset, validation_data=valid_dataset, epochs=50, callbacks=[earlystopping, reduce_lr])

import matplotlib.pyplot as plt
import seaborn as sns

loss = history.history['loss']
val_loss = history.history['val_loss']
figure = plt.figure(figsize=(8, 8))
plt.plot(history.epoch, loss, label='Training Loss')
plt.plot(history.epoch, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('LOSS')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
figure = plt.figure(figsize=(8, 8))
plt.plot(history.epoch, acc, label='Training Accuracy')
plt.plot(history.epoch, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')

model.save("cnn_model.h5")