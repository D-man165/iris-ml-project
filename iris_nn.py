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
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam

import pandas as pd

print ('py ',sys.version)
print ('pd ',pd.__version__)
print ('sklearn ',sklearn.__version__)
print ('tf ',tf.__version__)
#exit()

# read in panda from csv file
df = pd.read_csv("/Users/diptimanbora/Library/CloudStorage/GoogleDrive-diptiman@arizona.edu/My Drive/iris.csv", header=0)

print (df.columns)
print (df.shape)
print (df.iloc[0,:])

df = pd.get_dummies(df, columns=['Species', ]) # creates dummy columns for presence of the category
df['Species_Iris-setosa'] = df['Species_Iris-setosa'].replace({True: 1., False:0.})
df['Species_Iris-versicolor'] = df['Species_Iris-versicolor'].replace({True: 1., False: 0.})
df['Species_Iris-virginica'] = df['Species_Iris-virginica'].replace({True: 1.,False: 0.})
print (df.dtypes)

#x = df.values[:,1:5]
#y = df.values[:,5:8]
x = df.iloc[:,1:5] # feature matrix: contains the information on factors
y = df.iloc[:,5:8] # target matrix: contains the categories: dummy matrices
print (type(x))
print (type(y))
print (x.shape)
print (y.shape)
print (x.iloc[0,:])
print (y.iloc[0,:])
#print (x[0,:])
#print (y[0,:])
#exit()
"""
iris_data = load_iris()

# Extract the feature matrix (x) and target vector (y)
x = iris_data.data
y = iris_data.target

# Convert the target vector to a one-hot encoded format
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1)) # reshape as column vector

# Convert to DataFrame for consistency with the rest of your code
x = pd.DataFrame(x, columns=iris_data.feature_names)
y = pd.DataFrame(y, columns=encoder.categories_[0])
"""

# Split the data for training and testing
#train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, shuffle=True)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, random_state=42)
nfeatures = train_x.shape[1]
print (nfeatures)
# test_y will be measured against predicted values from the model from test_x

# Build the model

model = Sequential()
model.add(Dense(10, input_shape=(nfeatures,), activation='relu', name='input'))
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