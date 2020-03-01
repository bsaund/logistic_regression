#!/usr/bin/env python

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np

import IPython


def make_binary_features(matrix):
    means = np.mean(matrix,axis=0)
    for j in range(len(means)):
        for i in range(matrix.shape[0]):
            matrix[i,j] = (matrix[i,j] > means[j]) * 1.0

    return matrix


penalty = 'l2'


X, y = load_iris(return_X_y=True)

y = [np.clip(n, 0, 1) for n in y]
X = np.array(X)
X = make_binary_features(X)
X_aug = np.concatenate([X, X[:,-1:]], axis=1)  


# clf = LinearRegression(penalty=penalty).fit(X, y)
# clf_aug = LinearRegression(penalty=penalty).fit(X_aug, y)
clf = LogisticRegression(random_state=0, penalty=penalty).fit(X, y)
clf_aug = LogisticRegression(random_state=0, penalty=penalty).fit(X_aug, y)

print(clf.coef_)
print(clf_aug.coef_)



IPython.embed()
