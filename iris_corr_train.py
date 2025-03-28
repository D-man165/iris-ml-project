from iris_nn import *
import matplotlib.pyplot as plt

df, train_x, test_x, test_y, pred_proba, history = main()

# correlation: train_x input

import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = train_x.corr()

plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Train Data', fontsize=15, pad=20)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()
# plt.savefig("iris_corr_train.pdf")