from iris_nn import *
import matplotlib.pyplot as plt
import pandas as pd

df, train_x, test_x, test_y, pred_proba, history = main()

# temporary test for 1st category
temp_test_y1 = test_y.iloc[:,0]
temp_pred_proba1 = pred_proba[:,0]

# print ("shape of test_y", temp_test_y.shape)
# print ("shape of pred_proba", temp_pred_proba.shape)
temp_pred_zero = np.zeros(temp_test_y1.shape[0])
ns_auc1 = roc_auc_score(temp_test_y1, temp_pred_zero)
lr_auc1 = roc_auc_score(temp_test_y1, temp_pred_proba1)
print ("auc score for 1st category", lr_auc1)

# ns: no skill
# ns plot is a straight line
# ls: logistic regression -> fit into output for classification
ns_fpr1, ns_tpr1, _ = roc_curve(temp_test_y1, temp_pred_zero)
lr_fpr1, lr_tpr1, _ = roc_curve(temp_test_y1, temp_pred_proba1)

# 2nd/all
# temporary test for 2nd category
temp_test_y2 = test_y.iloc[:,1]
temp_pred_proba2 = pred_proba[:,1]

ns_auc2 = roc_auc_score(temp_test_y2, temp_pred_zero)
lr_auc2 = roc_auc_score(temp_test_y2, temp_pred_proba2)
print ("auc score for 2nd category", lr_auc2)

ns_fpr2, ns_tpr2, _ = roc_curve(temp_test_y2, temp_pred_zero)
lr_fpr2, lr_tpr2, _ = roc_curve(temp_test_y2, temp_pred_proba2)

# 3rd/all
# temporary test for 3rd category
temp_test_y3 = test_y.iloc[:,1]
temp_pred_proba3 = pred_proba[:,1]
temp_pred_zero = np.zeros(temp_test_y3.shape[0])

ns_auc3 = roc_auc_score(temp_test_y3, temp_pred_zero)
lr_auc3 = roc_auc_score(temp_test_y3, temp_pred_proba3)
print ("auc score for 3rd category", lr_auc3)

ns_fpr3, ns_tpr3, _ = roc_curve(temp_test_y3, temp_pred_zero)
lr_fpr3, lr_tpr3, _ = roc_curve(temp_test_y3, temp_pred_proba3)

# plot the curves for the model
fig, ax = plt.subplots(3,2)
ax[0,0].plot(history.history['accuracy'],'b',label='training accuracy')
ax[0,0].plot(history.history['val_accuracy'],'r',label='val_accuracy')
ax[0,0].set(title='training accuracy', xlabel='epoch', ylabel='accuracy')
ax[0,0].legend(['train', 'test'], loc='lower right')

ax[0,1].plot(history.history['loss'],'b',label='training loss')
ax[0,1].plot(history.history['val_loss'],'r',label='val_loss')
ax[0,1].set(title='training loss', xlabel='epoch', ylabel='loss')
ax[0,1].legend(['train', 'test'], loc='upper right')

# first category
ax[1,0].plot(ns_fpr1, ns_tpr1, linestyle='--', label='No Skill')
ax[1,0].plot(lr_fpr1, lr_tpr1, marker='.', label='Logistic')
ax[1,0].set(title='roc: category #1', xlabel='fpr', ylabel='tpr')
ax[1,0].legend(['no skill', 'logistic'], loc='lower right')

# second category
ax[1,1].plot(ns_fpr2, ns_tpr2, linestyle='--', label='No Skill')
ax[1,1].plot(lr_fpr2, lr_tpr2, marker='.', label='Logistic')
ax[1,1].set(title='roc: category #2', xlabel='fpr', ylabel='tpr')
ax[1,1].legend(['no skill', 'logistic'], loc='lower right')

# third category
ax[2,0].plot(ns_fpr3, ns_tpr3, linestyle='--', label='No Skill')
ax[2,0].plot(lr_fpr3, lr_tpr3, marker='.', label='Logistic')
ax[2,0].set(title='roc: category #3', xlabel='fpr', ylabel='tpr')
ax[2,0].legend(['no skill', 'logistic'], loc='lower right')

# Hide the empty subplot
fig.delaxes(ax[2, 1])

plt.tight_layout()
plt.show()