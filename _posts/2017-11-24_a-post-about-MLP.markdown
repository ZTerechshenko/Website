# Back-propagation of Errors and Multilayer Perceptron


## Introduction 

This post is the result of my homework for IST: 597 Foundations of Deep Learning class at Penn State.

In the first part, I will show how I develop softmax regression model and multilayear perceptron (MLP) to solve the XOR problem. In the second part, I will fit a two hidden layer MLP to a version of the IRIS dataset.

## XOR problem

XOR (Exclusive OR) is a classic dataset, where data points are not linearly separable. The XOR theory can be find here: http://home.agh.edu.pl/~vlsi/AI/xor_t/en/main.htm


Loading libraries


```python
import os 
import numpy as np
import scipy.sparse
import pandas as pd  
import matplotlib.pyplot as plt  
from copy import deepcopy
```

Setting default parameters for the plots


```python
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
```


```python
np.random.seed(927)
# Load in the data from disk
path = os.getcwd() + '/MLP/data/xor.dat'  
data = pd.read_csv(path, header=None) 
```

Here is how our data looks like:


```python
data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)
y = y.flatten()

```

Let's try softmax regression first.

### XOR & Softmax regression

Softmax Regression (Multinomial Logistic Regression) is a generalization of logistic regression used for multi-class classification.

In this particular case, we apply it a to a dataset X of dimension (N x d) and target Y converted to a matrix of one-hot encodings. One hot encodings can be described as a method to transform categorical features to a format that improves classification and prediction accuracy of machine learning algorithms.


```python
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

```


```python
def oneHotIt(y):
    N = y.shape[0]
    OHX = scipy.sparse.csr_matrix((np.ones(N), (y, np.array(range(N)))))
    OHX = np.array(OHX.todense()).T
    return OHX

```


```python
def predict(X,theta):
    W = theta[0]
    b = theta[1]
    scores = np.dot(X,W)+b
    probs = np.apply_along_axis(softmax, 1, scores)
    return (scores, probs)

```

We learn our parameters by minimizing negative log-likelihood. We also introduce a second term to our cost function that penalizes the magnitude of the model parameters. This second term is an example of regularization - an approach used to reduce the generalization error of learning algorithm. In other words, we use regularization to prevent the model overfitting.


```python
def computeCost(X,y,theta,reg):
    N = X.shape[0]
    y_mat = oneHotIt(y)
    probs = predict(X,theta)[1]
    loss = -np.sum(y_mat*np.log(probs))/N + (reg/2)*np.sum(theta[0]**2)   
    return loss

```


```python
We estimate the model parameters using the gradient descent. 
```


      File "<ipython-input-959-b82d84b61fa1>", line 1
        We estimate the model parameters using the gradient descent.
                  ^
    SyntaxError: invalid syntax




```python
def computeGrad(X, y, theta, reg):
    dL_db = 0
    dL_dw = 0
    N = X.shape[0]
    y_mat = oneHotIt(y)
    probs = predict(X,theta)[1] 
    ddy = (probs-y_mat)
    dL_dw = np.dot(X.T, ddy)/N + reg*theta[0] 
    dL_db = np.sum(ddy, axis=0)/N              
    nabla = (dL_dw, dL_db) 
    return nabla

```

Next, we initialize our parameters (weights and bias)


```python
# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1
```


```python
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))
theta = (W,b)
```

Finally, we specify our hyperparameters (number of epochs, learning rate, regularization strength) and train the model. These hyperparameters can be sucessfully tuned using Grid Search or Randomized Search. The difference between these 2 methods is that the random search approach samples hyperparameters from parameters' dictionary via a random, uniform distribution, while grid search trains and evaluates a machine learning classifier for each and every combination of hyperparameter values. In this example, I picked these values by hand.



```python
n_e = 1000
check = 100 # every so many pass/epochs, print loss/error to terminal
step_size = 0.01 #learning rate
reg = 0.01 # regularization strength
```

In order to calculate the gradients of the loss function with respect to our parameters (weights and bias), we use backpropagation algorithm. Backpropagation is a powerful technique that allows us to calculate our derivatives very efficiently. 


```python

cost = []
# gradient descent loop
num_examples = X.shape[0]
for i in range(n_e):

    loss = 0.0

    dL_dw, dL_db = computeGrad(X, y, theta, reg)
    b = theta[1]
    w = theta[0]
    new_b = b - (step_size * dL_db)
    new_w = w - (step_size * dL_dw)
    
    
    theta = (new_w, new_b)

    loss = computeCost(X, y, theta, reg)
    cost.append(loss)
    
    if i % check == 0:
        print("iteration %d: loss %f" % (i, loss))
```

    iteration 0: loss 0.693157
    iteration 100: loss 0.693154
    iteration 200: loss 0.693153
    iteration 300: loss 0.693151
    iteration 400: loss 0.693151
    iteration 500: loss 0.693150
    iteration 600: loss 0.693149
    iteration 700: loss 0.693149
    iteration 800: loss 0.693149
    iteration 900: loss 0.693148



```python
# evaluate training set accuracy
scores, probs = predict(X,theta)
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
```

    training accuracy: 0.50



```python
plt.plot(cost)
plt.title('Loss vs Epoch')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
```


![png](output_25_0.png)


It is obvious that multinomial logistic model does not fit the data well. 

Now we extend softmax regression model to an MLP using activation function ReLU. The activation function transforms the inputs of the layer into its outputs. Since we have a non-linearly separable problem, we choose a nonlinear activation function. ReLU (linear rectifier unit) is one of the most popular activations functions in deep learning due to its several mathematical advantages and good performance in general.

I also implement gradient-checking which is a method for numerically checking the derivatives to make sure that my implementation is correct (http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization).



```python

def relu(x):
    return np.maximum(x,0)


def oneHotIt(y):
    N = y.shape[0]
    #Y = Y[:,0]
    OHX = scipy.sparse.csr_matrix((np.ones(N), (y, np.array(range(N)))))
    OHX = np.array(OHX.todense()).T
    return OHX

def computeCost(X,y,theta,reg):
    N = X.shape[0]
    y_mat = oneHotIt(y)
    probs = predict(X,theta)[1]
    loss = -np.sum(y_mat*np.log(probs))/N +(reg/2)*(np.sum(theta[0]**2)+np.sum(theta[2]**2))  
    return loss



def computeNumGrad(X,y,theta,reg): # returns approximate nabla
    eps = 1e-5
    theta_list = deepcopy(list(theta))
    nabla_n = deepcopy(theta_list)
    ii = 0
    for param in theta_list:
        for i in range(param.shape[0]):
            for j in range(param.shape[1]):
                    param_p = deepcopy(param[i,j]) + eps
                    theta_p = deepcopy(theta_list)
                    theta_p[ii][i,j] = deepcopy(param_p)
                    loss_p = computeCost(X,y,theta_p,reg)
                    param_m = deepcopy(param[i,j]) - eps
                    theta_m = deepcopy(theta_list)
                    theta_m[ii][i,j] = deepcopy(param_m)
                    loss_m = computeCost(X,y,theta_m,reg)
                    param_grad = (loss_p - loss_m)/(2*eps)
                    nabla_n[ii][i,j]= deepcopy(param_grad)
        ii += 1
    return tuple(nabla_n)	


def computeGrad(X,y,theta,reg): # returns nabla
    W = theta[0]
    b = theta[1]
    W2 = theta[2]
    b2 = theta[3]
    
    N = X.shape[0]
    y_mat = oneHotIt(y)
    probs = predict(X, theta)[1] 
    h_pre = np.dot(X,W)+b         
    h = relu(h_pre) 
    h_act = deepcopy(h)
    h_act[h_act > 0] = 1             
    dh = np.dot(((probs-y_mat)/N), W2.T)
    dh_pre = np.multiply(dh, h_act)
  
    ddy = (probs-y_mat)
    
    db = np.sum(dh_pre, axis =0)
    dW = np.dot(X.T,dh_pre)+(reg*W)
    db2 = np.sum(ddy, axis = 0)/N
    dW2 = np.dot(h.T,(ddy/N)) + (reg*W2) 
    
    return (dW,db,dW2,db2)



def predict(X,theta):
    W = theta[0]
    b = theta[1]
    W2 = theta[2]
    b2 = theta[3]
    h_pre = np.dot(X,W)+b            
    h = relu(h_pre) 
    scores = np.dot(h,W2)+b2               
    probs = np.apply_along_axis(softmax, 1, scores)
    return (scores,probs)


# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1

# initialize parameters in such a way to play nicely with the gradient-check! 
h = 6 #100 # size of hidden layer
W = 0.05 * np.random.randn(D,h) #0.01 * np.random.randn(D,h)
b = np.zeros((1,h)) + 1.0
W2 = 0.05 * np.random.randn(h,K) #0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K)) + 1.0
theta = (W,b,W2,b2) 

# some hyperparameters
reg = 1e-3 # regularization strength

nabla_n = computeNumGrad(X,y,theta,reg)
nabla = computeGrad(X,y,theta,reg)
nabla_n = list(nabla_n)
nabla = list(nabla)

for jj in range(0,len(nabla)):
    is_incorrect = 0 # set to false
    grad = nabla[jj]
    grad_n = nabla_n[jj]
    err = np.linalg.norm(grad_n - grad) / (np.linalg.norm(grad_n + grad))
    if(err > 1e-8):
        print("Param {0} is WRONG, error = {1}".format(jj, err))
    else:
        print("Param {0} is CORRECT, error = {1}".format(jj, err))

# re-init parameters
h = 6 #100 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))
theta = (W,b,W2,b2) 

# some hyperparameters
n_e = 1000
check = 100 # every so many pass/epochs, print loss/error to terminal
step_size = 1e-0
reg = 0.01 # regularization strength
cost = []
# gradient descent loop
for i in range(n_e):
    loss = 0.0
    dL_dw, dL_db, dL_dW2, dL_db2 = computeGrad(X, y, theta, reg)
    W = theta[0]
    b = theta[1]
    W2 = theta[2]
    b2 = theta[3]
    new_b = b - (step_size * dL_db)
    new_W = W - (step_size * dL_dw)
    new_b2 = b2 - (step_size * dL_db2)
    new_W2 = W2 - (step_size * dL_dW2)
    theta = (new_W, new_b, new_W2, new_b2)
    loss = computeCost(X, y, theta, reg)
    cost.append(loss)
 
    if i % check == 0:
        print("iteration %d: loss %f" % (i, loss))
plt.plot(cost)
plt.title('Loss vs Epoch')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

scores, probs = predict(X,theta)
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))


```

    Param 0 is CORRECT, error = 1.2735902082075299e-09
    Param 1 is CORRECT, error = 7.006618457169421e-10
    Param 2 is CORRECT, error = 3.628690715693632e-11
    Param 3 is CORRECT, error = 3.524250349826579e-11
    iteration 0: loss 0.693158
    iteration 100: loss 0.528438
    iteration 200: loss 0.233728
    iteration 300: loss 0.235088
    iteration 400: loss 0.236514
    iteration 500: loss 0.228714
    iteration 600: loss 0.240271
    iteration 700: loss 0.230796
    iteration 800: loss 0.234034
    iteration 900: loss 0.234663



![png](output_27_1.png)


    training accuracy: 1.00


We achieved the accuracy of 100%!

## MLP for IRIS data

In this part of my blog, I will fit a 2-layer MLP to another classic dataset in machine learning - IRIS.


```python
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
```


```python
def predict(X,theta):
    W = theta[0]
    b = theta[1]
    W2 = theta[2]
    b2 = theta[3]
    W3 = theta[4]
    b3 = theta[5]
    h_pre1 = np.dot(X,W)+b            
    h1 = relu(h_pre1) 
    h_pre2 = np.dot(h1,W2)+b
    h2 = relu(h_pre2)
    scores = np.dot(h2,W3)+b3               
    probs = np.apply_along_axis(softmax, 1, scores)
    return (scores,probs)

```





```python
def computeGrad(X,y,theta,reg): # returns nabla
    W = theta[0]
    b = theta[1]
    W2 = theta[2]
    b2 = theta[3]
    W3 = theta[4]
    b3 = theta[5]
        
    N = X.shape[0]
    y_mat = oneHotIt(y)
    probs = predict(X, theta)[1]
    
    ddy = (probs-y_mat)
        
    h_pre1 = np.dot(X,W)+b         
    h1 = relu(h_pre1) 
    h_act1 = deepcopy(h1)
    h_act1[h_act1 > 0] = 1  
          
    h_pre2 = np.dot(h1,W2)+b2         
    h2 = relu(h_pre2) 
    h_act2 = deepcopy(h2)
    h_act2[h_act2 > 0] = 1  
          
    dh2 = np.dot((ddy/N), W3.T)
    dh_pre2 = np.multiply(dh2, h_act2)

      
    dh1 = np.dot(dh_pre2, W2.T)
    dh_pre1 = np.multiply(dh1, h_act1)

    db = np.sum(dh_pre1, axis =0)
    dW = np.dot(X.T,dh_pre1)+(reg*W)
    
    db2 = np.sum(dh_pre2, axis =0)
    dW2 = np.dot(h1.T,dh_pre2)+(reg*W2)
    
    db3 = np.sum(ddy, axis = 0)/N
    dW3 = np.dot(h2.T,ddy)/N + (reg*W3) 

    return (dW,db,dW2,db2,dW3,db3)
```

Here, instead of training the model on the full batch of data, we use mini-batches, which allow for a more robust convergence, avoiding local minima. Furthermore, with mini-batches we don't have to have all training data in memory and algorithm implementations.



```python
def create_mini_batch(X, y, start, end):
    mb_x = X[start:end]
    mb_y = y[start:end]
    return (mb_x, mb_y)
```


```python
def shuffle(X,y):
    ii = np.arange(X.shape[0])
    ii = np.random.shuffle(ii)
    X_rand = X[ii]
    y_rand = y[ii]
    X_rand = X_rand.reshape(X_rand.shape[1:])
    y_rand = y_rand.reshape(y_rand.shape[1:])
    return (X_rand,y_rand)

```


```python
np.random.seed(927)
# Load in the data from disk
path = os.getcwd() + '/MLP/data/iris_train.dat'  
data = pd.read_csv(path, header=None) 

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)
y = y.flatten()
```


```python
# load in validation-set
path = os.getcwd() + '/MLP/data/iris_test.dat'
data = pd.read_csv(path, header=None) 
cols = data.shape[1]  
X_v = data.iloc[:,0:cols-1]  
y_v = data.iloc[:,cols-1:cols] 

X_v = np.array(X_v.values)  
y_v = np.array(y_v.values)
y_v = y_v.flatten()
```


```python
# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1

h = 100 # size of hidden layer
h2 = 100 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,h2)
b2 = np.zeros((1,h2))
W3 = 0.01 * np.random.randn(h2,K)
b3 = np.zeros((1,K))
theta = (W,b,W2,b2,W3,b3)
```


```python
# some hyperparameters
n_e = 1000
n_b = 100
step_size = 0.01 #1e-0
reg = 0.001 # regularization strength
```


```python
train_cost = []
valid_cost = []
# gradient descent loop
num_examples = X.shape[0]
for i in range(n_e):
    X, y = shuffle(X,y) # re-shuffle the data at epoch start to avoid correlations across mini-batches
    loss = 0.0
    dL_dw, dL_db, dL_dW2, dL_db2, dL_dW3, dL_db3 = computeGrad(X, y, theta, reg)
    W = theta[0]
    b = theta[1]
    W2 = theta[2]
    b2 = theta[3]
    W3 = theta[4]
    b3 = theta[5]
    new_b = b - (step_size * dL_db)
    new_W = W - (step_size * dL_dw)
    new_b2 = b2 - (step_size * dL_db2)
    new_W2 = W2 - (step_size * dL_dW2)
    new_b3 = b3 - (step_size * dL_db3)
    new_W3 = W3 - (step_size * dL_dW3)
    theta = (new_W, new_b, new_W2, new_b2, new_W3, new_b3)
    loss = computeCost(X, y, theta, reg)
    train_cost.append(loss)
    s = 0
    while (s < num_examples):
        X_mb, y_mb = create_mini_batch(X,y,s,s + n_b)
        dL_dw, dL_db, dL_dW2, dL_db2, dL_dW3, dL_db3 = computeGrad(X, y, theta, reg)
        W = theta[0]
        b = theta[1]
        W2 = theta[2]
        b2 = theta[3]
        W3 = theta[4]
        b3 = theta[5]
        new_b = b - (step_size * dL_db)
        new_W = W - (step_size * dL_dw)
        new_b2 = b2 - (step_size * dL_db2)
        new_W2 = W2 - (step_size * dL_dW2)
        new_b3 = b3 - (step_size * dL_db3)
        new_W3 = W3 - (step_size * dL_dW3)
        theta = (new_W, new_b, new_W2, new_b2, new_W3, new_b3)
        loss = computeCost(X, y, theta, reg)
        valid_cost.append(loss)
        s += n_b

print(' > Training loop completed!')


scores, probs = predict(X,theta)
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: {0}'.format(np.mean(predicted_class == y)))

scores, probs = predict(X_v,theta)
predicted_class = np.argmax(scores, axis=1)
print('validation accuracy: {0}'.format(np.mean(predicted_class == y_v)))


```

     > Training loop completed!
    training accuracy: 0.9818181818181818
    validation accuracy: 0.975



```python
plt.plot(train_cost, label = 'Training')
plt.plot(valid_cost, label = 'Validation')

plt.legend(['Training', 'Validation'], loc='upper right')
plt.title('Loss vs Epoch')
plt.xlabel("Epoch")
plt.ylabel("Loss")
#plt.savefig('LossVsEpoch_2b.pdf')
plt.show()
```


![png](output_42_0.png)


We achieved a pretty high accuracy (97,5%) using MLP for IRIS data. 


# Conclusion

This exercise was pretty useful for understanding the differences between neural networks and regression model, the effect of hyperparameters on the model performance, as well as the importance of gradient checking in implementing even the simplest neural net model.


```python

```
