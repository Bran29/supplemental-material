import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
digits0 = datasets.load_digits(n_class=5)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

n_samples = len(digits0.images)
data0 = digits0.images.reshape((n_samples, -1))
# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)
X_train10=X_train[0:100]
y_train10=y_train[0:100]
X_test=X_test[0:500]
y_test=y_test[0:500]


fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(6, 6))
for idx, ax in enumerate(axs.ravel()):
    ax.imshow(X_train10[idx].reshape((8, 8)), cmap=plt.cm.binary)
    ax.axis("off")
_ = fig.suptitle("Subset1 from the 64-dimensional digits dataset", fontsize=16)

X_train0, X_test0, y_train0, y_test0 = train_test_split(
    data0, digits0.target, test_size=0.1, shuffle=False
)
X_train0=X_train0[0:400]
y_train0=y_train0[0:400]

fig, axs = plt.subplots(nrows=20, ncols=20, figsize=(6, 6))
for idx, ax in enumerate(axs.ravel()):
    ax.imshow(X_train0[idx].reshape((8, 8)), cmap=plt.cm.binary)
    ax.axis("off")

_ = fig.suptitle("Subset2 from the 64-dimensional digits dataset", fontsize=16)
# Learn the digits on the train subset

clf = svm.SVC(gamma=0.0001)
clf.fit(X_train10, y_train10)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)


disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix Under Dataset With Higher Cluster Diversity\n But Lower Data Diversity")
print(f"Confusion matrix:\n{disp.confusion_matrix}")


clf = svm.SVC(gamma=0.001)
clf.fit(X_train0, y_train0)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)


disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix Under Dataset With Higher Data Diversity\n But Lower Cluster Diversity")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
fig.savefig('mnist.eps',dpi=1200,format='eps')