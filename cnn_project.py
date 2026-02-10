import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
import pandas as pd
import matplotlib as plt
import numpy as np

#load data
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train.shape
x_test.shape
y_train[:5]
y_train = y_train.reshape(-1,)
#visualize
def plot_images(x, y, index):
    plt.figure(figsize=(10, 10))
    plt.imshow(x[index])
    plt.title(f"Label: {y[index]}")

plot_images(x_train, y_train,2)
#scale data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
#ANN
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Flatten(),
    layers.Dense(1300, activation='relu'),
    layers.Dense(800, activation='relu'),
    layers.Dense(300, activation='relu'),
    layers.Dense(10, activation='sigmoid')

])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2)

#evalution and repot of ann
from sklearn.metrics import confusion_matrix , classification_report
y_pred = model.predict(x_test)
# Convertir les probabilités en classes
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_flat = y_test.reshape(-1)
    
print(classification_report(y_test_flat, y_pred_classes))
#cnn model
model_cnn = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(1300, activation='relu'),
    layers.Dense(800, activation='relu'),
    layers.Dense(300, activation='relu'),
    layers.Dense(10, activation='sigmoid')

])
model_cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model_cnn.fit(x_train, y_train, epochs=2)

#evalution and repot of cnn
y_pred = model_cnn.predict(x_test)
# Convertir les probabilités en classes
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_flat = y_test.reshape(-1)
    
print(classification_report(y_test_flat, y_pred_classes))