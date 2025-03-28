import sys
sys.path.append('/Users/diptimanbora/Desktop/GitHub/iris-ml-project')

from iris_nn import *
import matplotlib.pyplot as plt

df, train_x, test_x, test_y, pred_proba, history = main()

# histogram plots: train_x input

# Filter the rows in df that are in test_x
train_indices = train_x.index
train_x_species = df.loc[train_indices]
# this is done because the dummy columns are not in test_x
# corresponding rows are pulled from df and stored in test_x_with_species

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
num_cols = 2
num_rows = 2
plt.figure(figsize=(10, 8))

for i, feature in enumerate(features):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.hist(train_x_species[train_x_species['Species_Iris-setosa'] == 1][feature], bins=20, color='red', alpha=0.5, label='Setosa')
    plt.hist(train_x_species[train_x_species['Species_Iris-versicolor'] == 1][feature], bins=20, color='blue', alpha=0.5, label='Versicolor')
    plt.hist(train_x_species[train_x_species['Species_Iris-virginica'] == 1][feature], bins=20, color='green', alpha=0.5, label='Virginica')
    plt.title(f'Histogram of {feature}', fontsize=18)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend(fontsize=12)

plt.suptitle('Histograms for Each Species: Train Data', fontsize=24)
plt.tight_layout()
plt.show()
# plt.savefig("iris_hist_train.pdf")