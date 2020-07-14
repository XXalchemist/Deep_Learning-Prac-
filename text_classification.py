'''
Text Classification using Tensorflow -
Classification of Movies through IMDB database 
'''

# Importing libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Loading dataset

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 10000 ) # Split 10000 words

# Getting word_index

word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}

print(word_index[0])

word_index['<PAD>'] = 0 # Creating Padding Word Tag, all comments have different length so we give padding tag so that each and every comment have same length
word_index['<START>'] = 1 # Creating Start Word Tag
word_index['<UNK>'] = 2 # Creating unknown Word Tag
word_index['<UNUSED>'] = 3 # Creating unused Word Tag


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) #Giving integer pointing towards words

# Preprocessing of data

# Allowing 150 words in comments

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index['<PAD>'], padding ='post', maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index['<PAD>'], padding ='post', maxlen = 250)

# Function For making prediction readable return text comment
def decode_review(test):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

print(decode_review(test_data[0]))

