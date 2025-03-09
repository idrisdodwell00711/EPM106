import numpy as np
import math
import struct 
import matplotlib as plt 
from sklearn import svm, metrics
import cv2
import tensorflow as tf
TF_ENABLE_ONEDNN_OPTS=0
   

class SVM:
    def __init__(self, learning_rate = 0.001, lamb_param = 0.01, n_iter= 3):
        self.lr = learning_rate
        self.lamb_param = lamb_param
        self.n_iter = n_iter
        self.w = None
        self.b = None
 
        
   
    def fit(self, X, y):
            print(X.shape)
            n_samples, n_features = X.shape
            
            #The multiclassifier requires an SVM for each number (one vs all)
            #The first svm there is 0 vs every other num- so it is either a 0 or 1-9
            y_ = np.where(y == 1, -1, 1)
            y_ = y_.reshape(-1, 1).astype(np.double)
           
            
 
            # The next svm will be 1 vs 2-9   
            
            # Intialize to create random weights
            self.w = np.random.rand(n_features)
            self.b = 0
            print(self.w.shape )
            
            for _ in range(self.n_iter):
                for idx, x_i in enumerate(X):
                    if(y_[idx]==float(1)):
                        condition = y_[idx] * (np.dot(x_i, self.w)-self.b) >= 1
                    
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
   
    def Gaussian_k(X, gamma):
          
          # Free parameter gamma
          if gamma == 'auto':
              gamma = 1.0/(X.shape[1] * X.var())
              
          x = X
        # get gaussian kernel Equation
          K = np.zeros((x.shape[0], X.shape[1]))
          
          # get gaussian kernel Equation
          #the range of the list is hard coded only for diagnostics, use x.shape[0] when code is functional
          for idx_i,_ in enumerate(range(20)):
            
            count_i = 0
            
            for idx_j, _ in enumerate(range(X.shape[1])):
                pixel_vector =np.array([X[idx_i][count_i], count_i])
     
                count_i = count_i +1
                
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
    for idx, _ in enumerate(range(test_rang)):
          flat_img = test_X[idx].flatten()
          flat_img = [1.0 if n > 127 else 0.0 for n in flat_img ]
          test_X_flat[idx] = flat_img
   
    train_X_flat = Gaussian_k(train_X_flat, gamma=0.01)
    clf = SVM()
    
    w, b = clf.fit(train_X_flat, train_y)
    predictions = clf.predict(test_X_flat)
    # test_y = np.where(test_y==1, -1, 1)
    
    print("SVM classification accuracy", accuracy(test_y, predictions))
    
 
    
    
    
    
    
   
   
    
    
