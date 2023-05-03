import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn import svm, datasets
# from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# class_names = iris.target_names

# # Split the data into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# # Run classifier, using a model that is too regularized (C too low) to see
# # the impact on the results

train = open('./image_pixels_train.csv', 'r')
test = open('./image_pixels_test.csv', 'r')

X_train = train['pixels']
X_test = test['pixels']

y_train = train['emotion']
y_test = test['emotion']

class_names = ['emotion']

classifier = open('./pretrained_ckpt', 'r')

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
