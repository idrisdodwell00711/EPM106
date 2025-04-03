import numpy as np                  # for basic operations over arrays
from scipy.spatial import distance  # to compute the Gaussian kernel
import cvxopt                       # to solve the dual optimization problem
import copy                         # to copy numpy arrays 

import numpy as np
import math
import struct 
import matplotlib as plt 
from sklearn import svm, metrics
import cv2
import tensorflow as tf
TF_ENABLE_ONEDNN_OPTS=0


class SVM:
    linear = lambda x, xࠤ , c=0: x @ xࠤ .T
    polynomial = lambda x, xࠤ , Q=5: (1 + x @ xࠤ.T)**Q
    rbf = lambda x, xࠤ , γ=0.01: np.exp(-γ * distance.cdist(x, xࠤ,'sqeuclidean'))
    kernel_funs = {'linear': linear, 'polynomial': polynomial, 'rbf': rbf}
    
    def __init__(self, kernel='rbf', C=1, k=2):
        # set the hyperparameters
        self.kernel_str = kernel
        self.kernel = SVM.kernel_funs[kernel]
        self.C = C                  # regularization parameter
        self.k = k                  # kernel parameter
        
        # training data and support vectors
        self.X, y = None, None
        self.αs = None
        
        # for multi-class classification
        self.multiclass = False
        self.clfs = []                                  

# This is useless here (only for notebook)
SVMClass = lambda func: setattr(SVM, func.__name__, func) or func


@SVMClass
def fit(self, X, y, eval_train=False):
    if len(np.unique(y)) > 2:
        self.multiclass = True
        return self.multi_fit(X, y, eval_train)
    
    # relabel if needed
    if set(np.unique(y)) == {0, 1}: y[y == 0] = -1
    # ensure y has dimensions Nx1
    self.y = y.reshape(-1, 1).astype(np.double) # Has to be a column vector
    self.X = X
    N = X.shape[0]
    
    # compute the kernel over all possible pairs of (x, x') in the data
    #self.K = self.kernel(X, X, self.k)
    
    # For 1/2 x^T P x + q^T x
    P = cvxopt.matrix(self.y @ self.y.T)
    q = cvxopt.matrix(-np.ones((N, 1)))
    
    # For Ax = b
    A = cvxopt.matrix(self.y.T)
    b = cvxopt.matrix(np.zeros(1))

    # For Gx <= h
    G = cvxopt.matrix(np.vstack((-np.identity(N),
                                 np.identity(N))))
    h = cvxopt.matrix(np.vstack((np.zeros((N,1)),
                                 np.ones((N,1)) * self.C)))

    # Solve    
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    self.αs = np.array(sol["x"])
        
    # Maps into support vectors
    self.is_sv = ((self.αs > 1e-3) & (self.αs <= self.C)).squeeze()
    self.margin_sv = np.argmax((1e-3 < self.αs) & (self.αs < self.C - 1e-3))
    
    if eval_train:  
      print(f"Finished training with accuracy {self.evaluate(X, y)}")


@SVMClass
def multi_fit(self, X, y, eval_train=False):
    self.k = len(np.unique(y))      # number of classes
    # for each pair of classes
    for i in range(self.k):
        # get the data for the pair
        Xs, Ys = X, copy.copy(y)
        # change the labels to -1 and 1
        Ys[Ys!=i], Ys[Ys==i] = -1, +1
        # fit the classifier
        clf = SVM(kernel=self.kernel_str, C=self.C, k=self.k)
        clf.fit(Xs, Ys)
        # save the classifier
        self.clfs.append(clf)
    if eval_train:  
      print(f"Finished training with accuracy {self.evaluate(X, y)}")

@SVMClass
def predict(self, X_t):
    if self.multiclass: return self.multi_predict(X_t)
    xₛ, yₛ = self.X[self.margin_sv, np.newaxis], self.y[self.margin_sv]
    αs, y, X= self.αs[self.is_sv], self.y[self.is_sv], self.X[self.is_sv]

    b = yₛ - np.sum(αs * y * self.kernel(X, xₛ, self.k), axis=0)
    score = np.sum(αs * y * self.kernel(X, X_t, self.k), axis=0) + b
    
    return np.sign(score).astype(int)

@SVMClass
def multi_predict(self, X):
    # get the predictions from all classifiers
    preds = np.zeros((X.shape[0], self.k))
    for i, clf in enumerate(self.clfs):
       preds[:, i] = clf.predict(X)
    
    # get the argmax and the corresponding score
    return np.argmax(preds, axis=1)

@SVMClass
def evaluate(self, X,y): 
    outputs = self.predict(X)
    accuracy = np.sum(outputs == y) / len(y)
    return round(accuracy, 2)


from sklearn.datasets import make_classification
import numpy as np

# Load the dataset
np.random.seed(1)
# X, y = make_classification(n_samples=500, n_features=2, 
#           n_redundant=0, n_informative=2, n_classes=4, 
#           n_clusters_per_class=1,  class_sep=0.3)

mnist = tf.keras.datasets.mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()


train_X = train_X.astype('float32') / 255
test_X = test_X.astype('float32') / 255

def resize(mnist):
   train_data = []
   for img in mnist:
          resized_img = cv2.resize(img, (14, 14))
          train_data.append(resized_img)
   return train_data
train_X = np.array(resize(train_X) ) 

test_X = np.array(resize(test_X))

train_X_flat = np.zeros((train_X.shape[0], train_X.shape[1]*train_X.shape[2]))
test_X_flat = np.zeros((test_X.shape[0], test_X.shape[1]*test_X.shape[2]))

rang = train_X.shape[0]
test_rang = test_X.shape[0]

for idx, _ in enumerate(range(rang)):
      flat_img = train_X[idx].flatten()
      if idx == 0:
          print(flat_img, len(flat_img))
      
      flat_img = [1.0 if n > (127/255) else 0.0 for n in flat_img ]
      if idx == 0:
          print(flat_img, len(flat_img), train_y[idx])
      train_X_flat[idx] = flat_img

# Test SVM
svm = SVM(kernel='rbf', k=4)
svm.fit(train_X_flat, train_y, eval_train=True)

y_pred = svm.predict(test_X_flat)
print(f"Accuracy: {np.sum(train_y==y_pred)/train_y.shape[0]}")


# Test with Scikit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

clf = OneVsRestClassifier(SVC(kernel='rbf', C=1, gamma=0.01)).fit(train_X_flat, train_y)
y_pred = clf.predict(train_X_flat)
print(f"Accuracy: {sum(train_y==y_pred)/train_y.shape[0]}", 'OvA')