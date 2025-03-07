import numpy as np
import math
import struct 
import matplotlib as plt 
from sklearn import svm, metrics
import cv2
import tensorflow as tf
TF_ENABLE_ONEDNN_OPTS=0
   

class SVM:
    def __init__(self, learning_rate = 0.001, lamb_param = 0.01, n_iter= 5):
        self.lr = learning_rate
        self.lamb_param = lamb_param
        self.n_iter = n_iter
        self.w = None
        self.b = None
   
   
    def fit(self, X, y):
            n_samples, n_features = X.shape
            
            #The multiclassifier requires an SVM for each number (one vs all)
            #The first svm there is 0 vs every other num- so it is either a 0 or 1-9
            y_ = np.where(y == 0, -1, 1)
 
            # The next svm will be 1 vs 2-9   
            
            # Intialize to create random weights
            self.w = np.random.rand(n_features)
            self.b = 0
            print(self.w.shape)
            
            for _ in range(self.n_iter):
                for idx, x_i in enumerate(X):
                    if(y_[idx]==float(1)):
                        #comparing 8 vs 9
                        condition = y_[idx] * (np.dot(x_i, self.w)-self.b) >= 1
                    
                    #comparing the other numbers
                    else:
                        None
                    if(condition):
                        self.w -= self.lr*(2 * self.lamb_param * self.w)
                    else:
                        self.w -= self.lr*(2 * self.lamb_param * self.w - np.dot(y_[idx], x_i))
                        self.b -= self.lr * y_[idx]
            return self.w, self.b             

                        
    def predict(self, X):
            approx = np.dot(X, self.w) - self.b
            
            return np.sign(approx)
        
# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    from sklearn.datasets import make_circles
    
    def Gaussian_k(X, gamma):
          
          # Free parameter gamma
          if gamma == 'auto':
              gamma = 1.0/(X.shape[1] * X.var())
              
          x = X
        # get gaussian kernel Equation
          K = np.zeros((x.shape[0], X.shape[1]))
          
          # get gaussian kernel Equation
          #the range of the list is hard coded only for diagnostics, use x.shape[0] when code is functional
          for idx_i,_ in enumerate(range(30)):
            
            count_i = 0
            
            for idx_j, _ in enumerate(range(X.shape[1])):
                pixel_vector =np.array([X[idx_i][count_i], count_i])
                
                
                # print(idx_j, count_i, 'pixel')
                count_i = count_i +1
                # summ = 0
                
            for idx_k, _ in enumerate(range(X.shape[1])):
                if count_i-1 != idx_k:
                        comparison_pixel_vector = np.array([X[idx_i][idx_k], idx_k])
                        
                        summ = sum(gamma*(pixel_vector - comparison_pixel_vector)**2)
                        K[idx_i, idx_j] = np.exp(-summ)
          print(len(K[0]))       
          return K
 
    
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    
    
    mnist = tf.keras.datasets.mnist

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    
    print(train_X.shape)
    n_train_samples, tr_x, tr_y = train_X.shape
    n_test_samples, t_x, t_y = test_X.shape
    
    test_X = test_X.reshape(n_test_samples, t_x, t_y)
    train_X = train_X.reshape(n_train_samples, tr_x, tr_y)
    
    train_X_flat = np.zeros((train_X.shape[0], train_X.shape[1]*train_X.shape[2]))
    test_X_flat = np.zeros((test_X.shape[0], test_X.shape[1]*test_X.shape[2]))
    
    rang = train_X.shape[0]
    test_rang = test_X.shape[0]

    for idx, _ in enumerate(range(rang)):
          flat_img = train_X[idx].flatten()
          train_X_flat[idx] = flat_img
    for idx, _ in enumerate(range(test_rang)):
          flat_img = test_X[idx].flatten()
          test_X_flat[idx] = flat_img
   
    train_X = Gaussian_k(train_X_flat, gamma=0.01)
    clf = SVM()
    
    w, b = clf.fit(train_X, train_y)
    predictions = clf.predict(test_X_flat)
    test_y = np.where(test_y==0, -1, 1)
    print("SVM classification accuracy", accuracy(test_y, predictions))
    
 
    
    
    
    
    
   
   
    
    
