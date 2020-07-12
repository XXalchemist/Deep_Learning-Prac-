'''
Introduction To Deep Learning with Python.
'''
# Prediction of clothes through images (shirt, shoes etc).

# Importing Libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Loading data(fashion includes different clothes) from keras

data = keras.datasets.fashion_mnist

# Splitting the images and labels into train and test data.

(train_images, train_labels), (test_images, test_labels) = data.load_data()
print(train_labels[6])

# Adding labels in list so that it can be easilyi identified later

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle-Boot']

# Showing the image using matplot

plt.imshow(train_images[7], cmap = plt.cm.binary)
plt.show()

# Creating a model

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)), # Input Layer
    keras.layers.Dense(128, activation= 'relu'), # Hidden Layer
    keras.layers.Dense(10, activation='softmax') # Output Layer
])

model.compile(optimizer='adam', loss ='sparse_categorical_crossentropy', metrics = ['accuracy'])

# Training of Model

model.fit(train_images, train_labels, epochs=5)


# Evaluation of model(testing of data)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested Acc : ", test_acc)

# Using model to predict

model.save('fashion_mnist.h5')
#load_model('fashion_mnist.h5')
prediction = model.predict([test_images]) # takes np.array()
print(np.argmax(prediction[0]))           # print the largest value and get the index of that value of image -> 0
print('Output : ',class_names[np.argmax(prediction[0])]) # Print the class name of the given result

# To present prediction for 5 inputs using matplot

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmp = plt.cmap.binary)
    plt.xlabel("Actual : ",class_labels[test_labels[i]])
    plt.title("Prediction : ", class_names[np.argmax(prediction[i])])
    plt.show()