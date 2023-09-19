# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model

![1](https://github.com/SaiDarshan2003/mnist-classification/assets/94692595/ed796099-4e8f-467f-8c20-fce7c9724ab0)


## DESIGN STEPS:

### STEP 1:
Import tensorflow and preprocessing libraries
### STEP 2:
Build a CNN model
### STEP 3:
Compile and fit the model and then predict

## PROGRAM
```
Developed by: Sai Darshan G
Register no : 212221240047
```
### Libraries:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
```
### Shaping:
```
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
```
### One Hot Encoding Outputs:
```
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
```
### Reshape Inputs:
```
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
```
### Build CNN Model:
```
model = keras.Sequential()
# Write your code here
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),activation='relu'))
model.add(layers.Conv2D(filters=64,kernel_size=(5,5),strides=(1,1),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(20,activation='relu'))
model.add(layers.Dense(15,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
```
### Compile and Fitting:
```
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
```
### Metrics
```
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
```
### Predict for own handwriting:
```
img = image.load_img('imagefive.jpeg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
```

## OUTPUT:

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/SaiDarshan2003/mnist-classification/assets/94692595/7db3b7b0-8029-4283-b03b-cf08cb3eba72)

### Classification Report
![image](https://github.com/SaiDarshan2003/mnist-classification/assets/94692595/6d6dfc1f-1d0a-46b0-b0b7-c4ede1b1cdf1)

### Confusion Matrix
![image](https://github.com/SaiDarshan2003/mnist-classification/assets/94692595/ec8e8e00-e9e7-4c65-88c7-f28220884f43)

### New Sample Data Prediction
![image](https://github.com/SaiDarshan2003/mnist-classification/assets/94692595/d7c3f690-5fa5-4fd0-9fc0-56d3225b530d)

![image](https://github.com/SaiDarshan2003/mnist-classification/assets/94692595/73595e97-1e99-4f3b-8b2c-160fd065d709)


## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
