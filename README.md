# **Iris-ml-project**
This is my first in a series of machine learning and data-science projects in Python. This is a basic ML project which classifies the famed Iris flower into 3 species(target variable) with 4 categories(feature variable). This project uses ML tools such as Scikit-Learn and additional modules such as TensorFlow, Adam and Matplotlib. I did not plan to use a neural network as my project, but in first-year college, as a research intern I had to build a simple neural network. Following projects will follow a "simple" to "complex" curve.

Originally this project was implemented using Jupyter Notebooks.

## Workings:
This project includes file `iris_nn.py` for loading the local iris dataset as a DataFrame, manipulating it and building and training the model. It also includes several other files which produce statistics and plots for the neural network, such as input histograms, correlation plots and kernel density plots (predicted probability plots).

### `iris_nn.py`:

#### Functions:
It consists of 3 functions `load_data()`, `build_model()` and `train_evaluate()`.
`load_data()` creates a DataFrame and then dummy columns for each species having 1 or 0 for boolean values. This is called [One Hot Encoding](https://www.geeksforgeeks.org/ml-one-hot-encoding/). Using train_test_split() from Scikit-Learn, it then obtains the testing and training feature and target matrices.

`build_model` creates a model using Sequential() from Keras with a input layer, a hidden layer consisting of 10 neurons with ReLU and an output layer with 3 neurons and Softmax. ReLU and Softmax are activation functions which introduce non-linear behavior for activations of neurons. Essentially activation functions decide the value of the activation for a neuron as a function of weights, biases and activations of the previous layer(which are then decided on the previous layer until the input). Softmax is generally used in output layer as it gives probability for each output which add up to 1. The function then selects Adam as its optimizer with a learning rate of 0.001. Visit the YouTube channel for an in-depth explanation [3Blue1Brown](https://www.youtube.com/@3blue1brown). The model is finally created here.

`train_and_evaluate()` runs the training set on the model which prints the loss and accuracy for each epoch(repetition). The `main()` runs all of these functions together. It also obtains the array of predicted probabilities for the testing set.

### Other files:
Other python files are in folder `plots`.
Other files and functions are quite simple and hence are not explained here. They deal with plots(which are not required, but nice to have) using Matplotlib. They display the ROC and AUC curves, correlation matrices, histogram plots and kernel density plots(predicted probability plots).

## To run the program:
If you simply want to see the neural network in action, simple run `iris_nn.py`. It will print out the accuracy and loss for each epoch as well as final accuracy and loss. To obtain the plots you do not need to run `iris_nn.py` separately. Simply run the file itself.

## So What?
* This project is not the best way to implement a neural network. I would argue the best way is to implement a neural network from scratch, i.e., without using Keras, Scikit-Learn or other modules. However, I was required to implement this on a time-basis and hence chose the easier way out.
  
* I learnt a ton from project, manipulating DataFrames, array, experimenting with different model configurations. And the plots were a ton of work too. I think adding the plots was a very smart move, as while experimenting on the plots I heightened my understanding of neural networks. In the future I aim to focus on basics, such as Linear Regression.

## About The Author:
 Here is some information about me:

* Name: Diptiman Bora
* Affiliation: BS in Physics and SDS, University of Arizona
* Location: Tucson, US.
* Current Role: Freshman at the University of Arizona and an undergraduate research assistance in UA.
* Check out my [LinkedIn Page](www.linkedin.com/in/diptiman-bora9724286).
