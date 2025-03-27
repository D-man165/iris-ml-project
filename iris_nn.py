import numpy as np
import sys

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

def load_data():
    df = pd.read_csv("/Users/diptimanbora/Library/CloudStorage/GoogleDrive-diptiman@arizona.edu/My Drive/iris.csv", header=0)
    df = pd.get_dummies(df, columns=['Species', ]) # creates dummy columns for presence of the category
    df['Species_Iris-setosa'] = df['Species_Iris-setosa'].replace({True: 1., False:0.})
    df['Species_Iris-versicolor'] = df['Species_Iris-versicolor'].replace({True: 1., False: 0.})
    df['Species_Iris-virginica'] = df['Species_Iris-virginica'].replace({True: 1.,False: 0.})

    x = df.iloc[:,1:5] # feature matrix: contains the information on factors
    y = df.iloc[:,5:8] # target matrix: contains the categories: dummy matrices

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, random_state=42)
    return df, train_x, test_x, train_y, test_y

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

def train_and_evaluate(model, train_x, train_y, test_x, test_y, batch_size=5, epochs=100):
    # batch size is the number of samples processed before model is updated: feature of SGD
    # epoch is number of complete passes through the data set: repetitions of the dataset
    print('batch_size, epochs = ', batch_size, epochs)
    history = model.fit(train_x, train_y, verbose=2, batch_size=batch_size, epochs=epochs, validation_data=(test_x, test_y))
    # contains the loss and accuracy for the training and validation data at each epoch

    results = model.evaluate(test_x, test_y)
    print()
    print('Final test set loss: {:4f}'.format(results[0]))
    print('Final test set accuracy: {:4f}'.format(results[1]))
    print(history.history.keys())
    return model, history

def main():
    df, train_x, test_x, train_y, test_y = load_data()
    nfeatures = train_x.shape[1]
    model, opt = build_model(nfeatures)

    trained_model, history = train_and_evaluate(model, train_x, train_y, test_x, test_y)
    pred_proba = trained_model.predict(test_x)

    # Return necessary data for plotting
    return df, test_x, test_y, pred_proba, history

if __name__ == "__main__":
    main()