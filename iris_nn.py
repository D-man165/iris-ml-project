import numpy as np
import sys
import matplotlib.pyplot as plt

import sklearn
from sklearn.datasets import load_iris  # useful for testing using sklearn's iris dataset
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

# Function to load and preprocess data
def load_data():
    iris_data = load_iris()
    x = iris_data.data
    y = iris_data.target

    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.reshape(-1, 1))  # reshape as column vector
    x = pd.DataFrame(x, columns=iris_data.feature_names)
    y = pd.DataFrame(y, columns=encoder.categories_[0])

    # Split the data for training and testing
    # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, shuffle=True)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, random_state=42)
    
    return train_x, test_x, train_y, test_y

# Function to build the model
def build_model(nfeatures):
    model = Sequential()
    model.add(Input(shape=(nfeatures,), name='input'))
    model.add(Dense(10, activation='relu', name='middle1'))
    model.add(Dense(3, activation='softmax', name='output'))  # Softmax turns the network's output into probabilities that sum to 1.

    # Adam optimizer with learning rate of 0.001
    opt = Adam(learning_rate=0.001)
    # example of stochastic gradient descent
    # breaks examples into mini batches to find out each step
    # significantly faster than batch gradient descent
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # Categorical Cross-Entropy then computes the loss based on how far 
    # off these probabilities are from the true distribution.

    return model, opt

# Function to train and evaluate the model
def train_and_evaluate(model, train_x, train_y, test_x, test_y, batch_size=5, epochs=100):
    # batch size is the number of samples processed before model is updated: feature of SGD
    # epoch is number of complete passes through the data set: repetitions of the dataset
    print('batch_size, epochs = ', batch_size, epochs)
    history = model.fit(train_x, train_y, verbose=2, batch_size=batch_size, epochs=epochs, validation_data=(test_x, test_y))
    # contains the loss and accuracy for the training and validation data at each epoch

    # Test the model
    results = model.evaluate(test_x, test_y)
    print()
    print('Final test set loss: {:4f}'.format(results[0]))
    print('Final test set accuracy: {:4f}'.format(results[1]))
    print(history.history.keys())

    return model, history

# Main function
def main():
    print('py ', sys.version)
    print('pd ', pd.__version__)
    print('sklearn ', sklearn.__version__)
    print('tf ', tf.__version__)

    train_x, test_x, train_y, test_y = load_data()
    nfeatures = train_x.shape[1]

    model, opt = build_model(nfeatures)

    print('optimizer parameters ')
    print(opt.get_config())

    print('neural network model summary ')
    print(model.summary())

    trained_model, history = train_and_evaluate(model, train_x, train_y, test_x, test_y)

    # obtain probabilities
    pred_proba = trained_model.predict(test_x)

# Ensure script runs when executed
if __name__ == "__main__":
    main()