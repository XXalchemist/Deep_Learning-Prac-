DEEP LEARNING
---
>pip install requirements.txt **Important** 

## Python Intro to Neural Network

1. What is Neural Network
2. Loading and Looking at Data
3. Creating a model in tensorflow
4. Using the model to make Prediction
5. Text Classification (example)

## What is Neural Network

Neural network is model that is used to recognise complex pattern in data. It is inspired by the biological neural network that constitutes human. It consist of multiple layers and every layer have two main attributes :- Nodes (as Neurons) and Edges (as the connection between the nodes).

![Neural_network_Diagram](Images/Neural_Networkpng.png)*Neural_Network*

**Some Examples :-**

#### Neural Network used for labelled data(Supervised Learning) :-
`Text Processing (RNTN)` `Image Recognition (CNN, DBN)` `Object Recognition (RNTN, CNN)` `Speech Recognition (RNN)`

#### Neural Network used for Unlabelled data(Unsupervised Learning) :-
`Feature Extraction (RBM)` `Pattern Recognition (Autoencoders)` 

## Loading and Looking at data 

```Python
# Prediction of clothes through images (shirt, shoes etc).

# Importing Libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

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
```

## Artificial Neural Network

_Python example_<br>


`Run ann.py`
