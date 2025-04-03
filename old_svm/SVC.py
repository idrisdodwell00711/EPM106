import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV 
from tensorflow.keras.datasets import mnist
import pickle

 

# Load and Visualize MNIST Dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

 
# def Gaussian_k(X, gamma):
#       print(X.shape)
      
#       squares = np.sum(X**2, axis=1) # x_i^2
#       squares_reshaped = squares.reshape(-1, 1) # x_j^2
#       dot_prod = np.dot(X, X.T) # x_i . x_j

#       sq_dists = squares_reshaped - 2 * dot_prod + squares

#       K = np.exp(-sum(gamma * sq_dists))
                    
#       print(len(K[0]))       
#       return K

# Visualize first 5 samples

# fig, axes = plt.subplots(1, 5, figsize=(10, 2))

# for i, ax in enumerate(axes):
#     ax.imshow(x_train[i], cmap='gray')
#     ax.set_title(f"Label: {y_train[i]}")
#     ax.axis('off')

# plt.show()

 

# 2. Preprocessing  8bit images coverion on 255 discerte points

x_train = x_train.astype(np.float32) / 255

x_test = x_test.astype(np.float32) / 255



def resize_images(images, size=(14, 14)):

    resized = [cv2.resize(img, size) for img in images]
    return np.array(resized)




x_train = resize_images(x_train)

x_test = resize_images(x_test)


 

x_train = x_train.reshape(x_train.shape[0], -1)

x_test = x_test.reshape(x_test.shape[0], -1)

for idx, img in enumerate(x_train):
      binary = np.array([1.0 if n > (127/255) else 0.0 for n in img ])
      x_train[idx] = binary
for idx, img in enumerate(x_test):
      binary = np.array([1.0 if n > (127/255) else 0.0 for n in img ])
      x_test[idx] = binary
 


# scaler = StandardScaler()

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)





# 3. Custom SVM Class

class SVM:

    def __init__(self, learning_rate, lambda_param, n_iters):

        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.loss_history = []
        self.acc_history = []

 

    def fit(self, X, y, t_X, t_y):

        n_samples, n_features = X.shape
        
        #init with random weights and bias at 0
        self.w = np.random.rand(n_features)
        self.b = 0

 

        for i in range(self.n_iters):

            loss = 0.01
            
            #change the loss function through the epochs
            third_iter = int(self.n_iters/3)
        
            
            three_quarts_iter = (self.n_iters//4)*3
            
            # if i == third_iter:
            #     self.lr = self.lr/20
            #     print('third', self.lr)
            # if i == three_quarts_iter:
            #     self.lr = self.lr/200
            #     print('3quarts', self.lr)

            for idx, x_i in enumerate(X):

                condition = y[idx] * (np.dot(x_i, self.w) - self.b) <= -1
                
                
                

                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)

                else:

                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

                    loss += 1 - y[idx] * (np.dot(x_i, self.w) - self.b)

            avg_loss = loss / n_samples
            #print('loss - ', avg_loss)

            self.loss_history.append(avg_loss)

            y_pred = self.predict(t_X)

            acc = accuracy_score(t_y, y_pred)
            #print('acc-',acc)

            self.acc_history.append(acc)

 

    def predict(self, X):
        print('weight',len(self.w))

        approx = np.dot(X, self.w) - self.b

        return np.sign(approx)

 

# 4. One-vs-Rest Multi-class Training

ovr_models = {}

digits = np.arange(10)

 

for digit in digits:


    print(f"\nTraining classifier for digit {digit} vs All...")

    y_train_binary = np.where(y_train == digit, -1, 1)
    y_test_binary = np.where(y_test == digit, -1, 1)
    print(digit)
    

    model = SVM(learning_rate=0.001, lambda_param=0.001, n_iters=2)
    
   
    model.fit(x_train, y_train_binary, x_test, y_test_binary)
    ovr_models[digit] = model
    
with open('ovr.pickle', 'wb') as file:
    pickle.dump(list(ovr_models.items()), file)  




# 5. Prediction for multiclass

def predict_multiclass(X):
    
   
    scores = []
    
    #Going over each number and its corresponding model
    with open('ovr.pickle', 'rb') as file:
        svms = pickle.load(file)
        print('svms', svms[0][0])

        for digit, model in svms:
    
            score = np.dot(X, model.w) - model.b
    
            scores.append(score)
        
        
    scores = np.array(scores)

    return np.argmax(scores, axis=0)

 

y_pred = predict_multiclass(x_test)

 

# 6. Evaluation

accuracy = accuracy_score(y_test, y_pred)

print(f"\nFinal Multi-class Accuracy: {accuracy * 100:.2f}%\n")

 

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

 

# 7. Visualization

# Plot Accuracy vs Iterations for each digit classifier

plt.figure(figsize=(12, 8))

for digit in digits:

    plt.plot(ovr_models[digit].acc_history, label=f'Digit {digit}')

plt.xlabel('Iterations')

plt.ylabel('Accuracy')

plt.title('Accuracy vs Iterations for Each Digit Classifier')

plt.legend()

plt.grid(True)

plt.show()

 

# Visualize one test image per digit 0-9

unique_digits = np.arange(10)

selected_indices = []

 

for digit in unique_digits:

    indices = np.where(y_test == digit)[0]

    if len(indices) > 0:

        selected_indices.append(indices[0])  # First occurrence of each digit

 

plt.figure(figsize=(10, 5))

for i, idx in enumerate(selected_indices):

    plt.subplot(2, 5, i + 1)

    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')

    plt.title(f"Pred: {y_pred[idx]} (True: {y_test[idx]})")

    plt.axis('off')

 

plt.tight_layout()

plt.show()
