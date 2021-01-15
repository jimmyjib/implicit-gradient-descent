# Implicit Gradient Descent

Backward(Implicit) Euler method is known to be more effective for stiff systems than forward Euler method. Since each neurons in network recognizes certain shape of data better than others(especially for image data), the linear system in each layer can very likely be a stiff system. 

The equation that represents basic gradient descent resembles forward Euler method. Although the fundamentals for the two equations are very different, it was worth to test whether the advantages of backward Euler method can also be seen in gradient descent. 

## Overview

- Change equation of gradient descent into implicit form
- Discover back-propagation method for **implicit gradient descent**
- Test the results and compare it with other optimization techniques 

## Equations

### 1. Implicit Gradient Descent

<p align="center">
    <img src="./Images/igd.PNG" width="500px">
</p>


Used conjugate gradient method to calculate difference. It requires back-propagation equations for the hessian matrices

### 2. Hessian

**1) Pre-actviation Values(Vector)**

<p align="center">
    <img src="./Images/hessian-z.PNG" width="500px">
</p>


**2) Weight Matrix**

<p align="center">
    <img src="./Images/hessian-w.PNG" width="500px">
</p>


## Results

- Used two networks. **Eve-master** and 2-layer fully connected network designed with numpy
- Used MNIST dataset for fully connected network
- Due to time complexity it was impossible to run the whole system with implicit gradient descent. Picked a few matrices and vectors to run with implicit form and the rest of the system was run with regular gradient descent

### 1. Eve-master

| learning rate = 0.01                                         | learning rate = 0.2                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="./Codes/Eve-master/eve_result/lr-0.01+maxiter-1000.png" width="400px"> | <img src="./Codes/Eve-master/eve_result/lr=0.2.png" width="400px"> |

### 2. FC network, MNIST dataset

**1) Implicit gradient descent on b1, b2**

| learning rate = 0.01                                         | learning rate = 0.2                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="./Codes/MNIST/mnist_result/db1+db2/lr-0.01.png" width="400px"> | <img src="./Codes/MNIST/mnist_result/db1+db2/lr=0.2.png" width="400px"> |

**2) Implicit gradient descent on b1, b2, w2**

| learning rate = 0.01                                         | learning rate = 0.2                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="./Codes/MNIST/mnist_result/db1+db2+w2/lr=0.01+epoch=100.png" width="400px"> | <img src="./Codes/MNIST/mnist_result/db1+db2+w2/lr=0.2+epoch=100.png" width="400px"> |

Overall, it showed better performance that other optimization methods, when learning rate was relatively high. It is expected to be effective when the system becomes much more complex and the initial value of learning rate is set to large value. 

## Reference

[Eve-master](https://github.com/Jeanselme/Eve)