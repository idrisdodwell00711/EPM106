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
   
        
    # def convert_num_to_bin(self,num):
    #     binary_num =[0,0,0,0]
    #     match num:
    #         case 0:
    #                   binary_num =[0,0,0,0]
    #         case 1:
    #                   binary_num =[0,0,0,1]
    #         case 2:
    #                   binary_num =[0,0,1,0]
    #         case 3:
    #                   binary_num =[0,0,1,1]
    #         case 4:
    #                   binary_num =[0,1,0,0]
    #         case 5:
    #                   binary_num =[0,1,0,1]
    #         case 6:
    #                   binary_num =[0,1,1,0]
    #         case 7:
    #                   binary_num = [0,1,1,1]
    #         case 8:
    #                   binary_num =[1,0,0,0]
    #         case 9:
    #                   binary_num =[1,0,0,1]
    #     return binary_num
   
    def fit(self, X, y):
            n_samples, nx_features, ny_features = X.shape
            
            # lst = [0,0,0,0]
            # arr = np.array(lst)
            # y_ = np.zeros((y.shape[0], arr.shape[0]))

            # for idx, i in enumerate(y):
            #     i_nt = int(i)
            #     y_[idx] = self.convert_num_to_bin(i_nt)
            
            
            #The multiclassifier requires an SVM for each number (one vs all)
            #The first svm there is 0 vs every other num- so it is either a 0 or 1-9
            y_ = np.where(y < 1, -1, 1)
 
            # The next svm will be 1 vs 2-9   
            
            # Intialize to create random weights
            self.w = np.random.rand(nx_features, ny_features)
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
              print(X.var(), 'h')
          x = X
        # get gaussian kernel Equation
          K = np.zeros((x.shape[0], X.shape[1], X.shape[2]))
          
          # get gaussian kernel Equation
          #the range of the list is hard coded only for diagnostics, use x.shape[0] when code is functional
          for idx_i,i in enumerate(range(1)):
            print(train_y[idx_i])
            count = 0
            for idx_j, j in enumerate(range(X.shape[1])):
                pixel = X[idx_i][idx_j][count]
                count = count+1
                summ = 0
                for idx_k, k in enumerate(range(X.shape[2])):
                      if count != k:
                          compare_pixel = X[idx_i][idx_j][idx_k]
                          summ += gamma*(pixel - compare_pixel)**2
                          K[i, j, k] = np.exp(-summ)
          print(K[0])              
          return K
 
    
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    
    
    mnist = tf.keras.datasets.mnist

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    
    
    n_train_samples, tr_x, tr_y = train_X.shape
    n_test_samples, t_x, t_y = test_X.shape
    
    test_X = test_X.reshape(n_test_samples, t_x, t_y)
    train_X = train_X.reshape(n_train_samples, tr_x, tr_y)
   
    train_X = Gaussian_k(train_X, gamma=0.001)
    clf = SVM()
    
    w, b = clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)

    print("SVM classification accuracy", accuracy(test_y, predictions))
    
 
    
    
    
    
    
   
   
    
    
