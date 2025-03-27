# example nn
# the aim is to take certain data/factors for different flowers and then predict its category

import numpy as np
import sys
import matplotlib.pyplot as plt

import sklearn
from sklearn.datasets import load_iris # usefull for testing using sklearn's iris dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

import pandas as pd

print ('py ',sys.version)
print ('pd ',pd.__version__)
print ('sklearn ',sklearn.__version__)
print ('tf ',tf.__version__)

iris_data = load_iris()
x = iris_data.data
y = iris_data.target

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1)) # reshape as column vector
x = pd.DataFrame(x, columns=iris_data.feature_names)
y = pd.DataFrame(y, columns=encoder.categories_[0])

# Split the data for training and testing
#train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, shuffle=True)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, random_state=42)
nfeatures = train_x.shape[1]

# Build the model

model = Sequential()
model.add(Input(shape=(nfeatures,), name='input'))
model.add(Dense(10, activation='relu', name='middle1'))
model.add(Dense(3, activation='softmax', name='output'))

# Adam optimizer with learning rate of 0.001
opt = Adam(learning_rate=0.001) 
# example of stochastic gradeint descent
# breaks examples into mini batches to find out each step
# significantly faster than batch gradient descent
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# Softmax turns the network's output into probabilities that sum to 1.
# Categorical Cross-Entropy then computes the loss based on how far 
# off these probabilities are from the true distribution.

print('optimizer parameters ')
print(opt.get_config())

print('neural network model summary ')
print(model.summary())

# Train the model
# batch size is the number of samples processed before model is updated: feature of SGD
# epoch is number of complete passes through the data set: repitions of the dataset
batch_size = 5
epochs = 100
print ('batch_size, epochs = ',batch_size,epochs)
history = model.fit(train_x, train_y, verbose=2, batch_size=batch_size, epochs=epochs, validation_data=(test_x,test_y))
# contains the loss and accuracy for the training and validation data at each epoch

# Test the model
results = model.evaluate(test_x, test_y)
print()
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))
print (history.history.keys())

# make probabilities
pred_proba = model.predict(test_x)