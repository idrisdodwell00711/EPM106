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
   
        
    def convert_num_to_bin(self,num):
        binary_num =[0,0,0,0]
        match num:
            case 0:
                      binary_num =[0,0,0,0]
            case 1:
                      binary_num =[0,0,0,1]
            case 2:
                      binary_num =[0,0,1,0]
            case 3:
                      binary_num =[0,0,1,1]
            case 4:
                      binary_num =[0,1,0,0]
            case 5:
                      binary_num =[0,1,0,1]
            case 6:
                      binary_num =[0,1,1,0]
            case 7:
                      binary_num = [0,1,1,1]
            case 8:
                      binary_num =[1,0,0,0]
            case 9:
                      binary_num =[1,0,0,1]
        return binary_num
   
    def fit(self, X, y):
            n_samples, nx_features, ny_features = X.shape
            
            lst = [0,0,0,0]
            arr = np.array(lst)
            y_ = np.zeros((y.shape[0], arr.shape[0]))

            for idx, i in enumerate(y):
                

                
                i_nt = int(i)
                
                y_[idx] = self.convert_num_to_bin(i_nt)
                
                
            
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
          print(X[0][6], 'x')
          # Free parameter gamma
          if gamma == 'auto':
              gamma = 1.0/(X.shape[1] * X.var())
          x = X
        # get gaussian kernel Equation
          K = np.zeros((x.shape[0], X.shape[1], X.shape[2]))
          
          # get gaussian kernel Equation
          for idx_i,i in enumerate(range(4)):
            count = 0
            
            for idx_j, j in enumerate(range(X.shape[1])):
                pixel = X[idx_i][idx_j][count]
                
                
                
                for idx_k, k in enumerate(range(X.shape[2])):
                      compare_pixel = X[idx_i][idx_j][idx_k]
                    
                   
                      # print(np.exp(-sum((pixel - compare_pixel)**2) * gamma))
                      K[i, j, k] = (np.exp(-(pixel - compare_pixel)**2 * gamma))
                     
                     
                count = count+1
          return K
    # def Gaussian_k(X, gamma):
    #       print(X)
    #       # Free parameter gamma
    #       if gamma == 'auto':
    #           gamma = 1.0/(X.shape[1] * X.var())
    #       x = X
          
    #     # get gaussian kernel Equation
    #       K = np.zeros((x.shape[0], X.shape[0]))
    #       # get gaussian kernel Equation
    #       for i in range(x.shape[0]):
    #         for j in range(X.shape[0]):
              
    #           K[i, j] = np.exp(-sum((X[i] - X[j])**2) * gamma)
    #       return K
    
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    
    
    mnist = tf.keras.datasets.mnist

    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    

    n_train_samples, tr_x, tr_y = train_X.shape
    n_test_samples, t_x, t_y = test_X.shape
    
    test_X = test_X.reshape(n_test_samples, t_x, t_y)
   

    train_X = train_X.reshape(n_train_samples, tr_x, tr_y)
    
    # y = np.where(y < 0, -1, 1)
    
    
    train_X = Gaussian_k(train_X, gamma=0.001)
    clf = SVM()
    
    w, b = clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)

    print("SVM classification accuracy", accuracy(test_y, predictions))
    
    # Creating dataset
    # X, y = datasets.make_blobs(
    #   n_samples = 10, # Number of samples
    #   n_features = 2, # Features
    #   centers = 2,
    #   cluster_std = 1.05,
    #   random_state=40)
    # # Classes 1 and -1
    

    
    
    
    
    
   
    # # X = Gaussian_k(X, gamma=0.085)
    
    # X_train, X_test, y_train, y_test = train_test_split(
    # X, y, test_size=0.2, random_state=123)
    
    
    # clf = SVM()
    
    # w, b = clf.fit(X_train, y_train)
    # predictions = clf.predict(X_test)

    # print("SVM classification accuracy", accuracy(y_test, predictions))


 

    

    # def visualize_svm():
    #     def get_hyperplane_value(x, w, b, offset):
    #         return (-w[0] * x + b + offset) / w[1]

    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #     plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

    #     x0_1 = np.amin(X[:, 0])
    #     x0_2 = np.amax(X[:, 0])

    #     x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    #     x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    #     x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    #     x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

    #     x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    #     x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

    #     ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    #     ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    #     ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    #     x1_min = np.amin(X[:, 1])
    #     x1_max = np.amax(X[:, 1])
    #     ax.set_ylim([x1_min - 3, x1_max + 3])

    #     plt.show()

    # visualize_svm()
    
    
