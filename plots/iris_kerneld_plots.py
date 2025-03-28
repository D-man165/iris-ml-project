import sys
sys.path.append('/Users/diptimanbora/Desktop/GitHub/iris-ml-project')

from iris_nn import *
import matplotlib.pyplot as plt

df, train_x, test_x, test_y, pred_proba, history = main()
# predicted probabilities and kernel density plots

import matplotlib.pyplot as plt
import seaborn as sns

pred_proba_df = pd.DataFrame(pred_proba, columns=["pred_Iris-setosa", "pred_Iris-versicolor", "pred_Iris-virginica"], index=test_y.index)
pred_proba_target = pd.concat([pred_proba_df, test_y], axis=1)

# Extract predicted probabilities for each category
pred_proba_target1 = np.array(pred_proba_target[pred_proba_target['Species_Iris-setosa'] == 1][["pred_Iris-setosa"]])
pred_proba_target_not1 = np.array(pred_proba_target[pred_proba_target['Species_Iris-setosa'] == 0][["pred_Iris-setosa"]])

pred_proba_target2 = np.array(pred_proba_target[pred_proba_target['Species_Iris-versicolor'] == 1][["pred_Iris-versicolor"]])
pred_proba_target_not2 = np.array(pred_proba_target[pred_proba_target['Species_Iris-versicolor'] == 0][["pred_Iris-versicolor"]])

pred_proba_target3 = np.array(pred_proba_target[pred_proba_target['Species_Iris-virginica'] == 1][["pred_Iris-virginica"]])
pred_proba_target_not3 = np.array(pred_proba_target[pred_proba_target['Species_Iris-virginica'] == 0][["pred_Iris-virginica"]])

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
plt.suptitle('Kernel Density Plots of Predicted Probabilities', fontsize=20)

# Plot for category 1
sns.kdeplot(pred_proba_target1.flatten(), fill=True, color='blue', label='Iris-setosa', common_norm=False, ax=axes[0, 0])
sns.kdeplot(pred_proba_target_not1.flatten(), fill=True, color='red', label='Not Iris-setosa', common_norm=False, ax=axes[0, 0])
axes[0, 0].set_title('Kernel Density Plot for Iris-setosa')
axes[0, 0].set_xlabel('Predicted Probability')
axes[0, 0].set_ylabel('Density')
axes[0, 0].grid(True)
axes[0, 0].legend()

# Plot for category 2
sns.kdeplot(pred_proba_target2.flatten(), fill=True, color='blue', label='Iris-versicolor', common_norm=False, ax=axes[0, 1])
sns.kdeplot(pred_proba_target_not2.flatten(), fill=True, color='red', label='Not Iris-versicolor', common_norm=False, ax=axes[0, 1])
axes[0, 1].set_title('Kernel Density Plot for Iris-versicolor')
axes[0, 1].set_xlabel('Predicted Probability')
axes[0, 1].set_ylabel('Density')
axes[0, 1].grid(True)
axes[0, 1].legend()

# Plot for category 3
sns.kdeplot(pred_proba_target3.flatten(), fill=True, color='blue', label='Iris-virginica', common_norm=False, ax=axes[1, 0])
sns.kdeplot(pred_proba_target_not3.flatten(), fill=True, color='red', label='Not Iris-virginica', common_norm=False, ax=axes[1, 0])
axes[1, 0].set_title('Kernel Density Plot for Iris-virginica')
axes[1, 0].set_xlabel('Predicted Probability')
axes[1, 0].set_ylabel('Density')
axes[1, 0].grid(True)
axes[1, 0].legend()

plt.delaxes(axes[1, 1])
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
plt.tight_layout()
plt.show()
# plt.savefig("iris_kerneld_plots.pdf")