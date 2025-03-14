import numpy as np
from numpy import ndarray
from typing import Callable
from contextlib import contextmanager

class Layer:
    def forward(self, _input: ndarray, train=True) -> ndarray:
        raise NotImplementedError("Forward propagation not implemented")

    def backward(self, prev_grad: ndarray, lr: float) -> ndarray:
        raise NotImplementedError("Backward propagation not implemented")

class WeightInitializer:
    @staticmethod
    def xavier_init(input_dim: int, output_dim: int) -> ndarray:
        limit = np.sqrt(6 / (input_dim + output_dim))
        return np.random.uniform(-limit, limit, size=(input_dim, output_dim))

    @staticmethod
    def he_init(input_dim: int, output_dim: int) -> ndarray:
        stddev = np.sqrt(2 / input_dim)
        return np.random.normal(0, stddev, size=(input_dim, output_dim))




class Activation(Layer):
    def __init__(self, func: Callable[[ndarray], ndarray], grad: Callable[[ndarray], ndarray]):
        self.func = func
        self.grad = grad
    
    def forward(self, _input: ndarray, train=True) -> ndarray:
        if train:
            self.input = _input
        return self.func(_input)
        
    def backward(self, prev_grad: ndarray, lr: float) -> ndarray:
        return prev_grad * self.grad(self.input)
        
class ReLU(Activation):
    def __init__(self):
        def func(x): return np.maximum(0, x)
        def grad(x): return np.where(x > 0, 1, 0)
        super().__init__(func,grad)
        
    def __repr__(self):
        return "ReLU()"
    
class Sigmoid(Activation):
    def __init__(self):
        def func(x): return 1 / (1 + np.exp(-x))
        def grad(x): return func(x) * (1 - func(x))
        super().__init__(func,grad)
    
    def __repr__(self):
        return "Sigmoid()"
        
class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, * , initializer: Callable[[int, int], ndarray] = WeightInitializer.he_init):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = initializer(input_size, output_size)
        self.bias = np.zeros((1,output_size))
        
    
    def forward(self, _input: ndarray, train=True) -> ndarray:
        if train:
            self.input = _input
        return _input.dot(self.weights) + self.bias
    
    def backward(self, prev_grad: ndarray, lr = float) -> ndarray:
        out_grad = prev_grad.dot(self.weights.T)
        self.weights -= lr * self.input.T.dot(prev_grad)
        self.bias -= lr * np.sum(prev_grad, axis=0, keepdims=True)
        return out_grad

    def __repr__(self):
        return f"Dense(input={self.input_size}, output={self.output_size})"

class Loss:
    def base(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        raise NotImplementedError("Base loss function not implemented")

    def grad(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        raise NotImplementedError("Loss gradient function not implemented")
    
    

class MSE(Loss):
    def base(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        return np.mean((y_true - y_pred)**2)
    
    def grad(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        return 2 * (y_pred - y_true) / np.size(y_true) 

    def __repr__(self):
        return "MSE()"


class BCE(Loss):
    def base(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        return y_true * np.log(y_pred + 1e-9) + (1-y_true) * np.log(1 - y_pred + 1e-9)
    
    def grad(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        return (y_pred - y_true) / np.size(y_true)
    def __repr__(self):
        return "BCE()"

class Optimizer:
    def __init__(self, layers: ndarray, lr: float):
        self.layers = layers[::-1]
        self.lr = lr
    def step(self, gradient: ndarray):
        raise NotImplementedError("Optimizer Step function not implemented")
    
class SGD(Optimizer):
    def __init__(self, layers: ndarray, lr: float = 0.01, momentum: float = 0.0):
        self.momentum = momentum
        super().__init__(layers, lr)
    
    def step(self, gradient):
        prev_grad = gradient
        for layer in self.layers:
            if isinstance(layer, Dense):
                if not hasattr(layer, "velocity"):
                    layer.velocity = np.zeros_like(layer.weights)
                layer.velocity = self.momentum * layer.velocity - self.lr * (1 - self.momentum) * layer.input.T.dot(prev_grad)
                out_grad = prev_grad.dot(layer.weights.T)
                layer.weights += layer.velocity
                layer.bias -= self.lr * np.sum(prev_grad, axis=0, keepdims=True)
            elif isinstance(layer, Activation):
                out_grad = prev_grad * layer.grad(layer.input)
            prev_grad = out_grad
    def __repr__(self):
        return f"SGD(lr={self.lr}, momentum={self.momentum})"
    
class RMSProp(Optimizer):
    def __init__(self, layers: ndarray, lr: float = 0.01, beta = 0.9):
        self.beta = beta
        super().__init__(layers, lr)
        
    def step(self, gradient):
        prev_grad = gradient
        for layer in self.layers:
            if isinstance(layer, Dense):
                if not hasattr(layer, "velocity"):
                    layer.velocity = np.zeros_like(layer.weights)
                
                layer.velocity = self.beta * layer.velocity + ( 1 - self.beta) * layer.weights ** 2
                layer.weights -= self.lr * layer.input.T.dot(prev_grad) / np.sqrt(layer.velocity + 1e-9)
                out_grad = prev_grad.dot(layer.weights.T)
            elif isinstance(layer, Activation):
                out_grad = prev_grad * layer.grad(layer.input)
            prev_grad = out_grad
    def __repr__(self):
        return f"RMSProp(lr={self.lr}, beta={self.beta})"
            
                

class NN:
    def __init__(self, layers: list[Layer], loss: Loss = MSE ,*, optimizer: Optimizer | None = None):
        self.layers = layers
        self.training = True
        self.loss: Loss = loss()
        self.optimizer: Optimizer | SGD = optimizer or SGD(self.layers)
    def forward(self, _input: ndarray) -> ndarray:
        output = _input
        for layer in self.layers:
            output = layer.forward(output, self.training)
        return output
    
    def backward(self, grad: ndarray) -> None:
        input_grad = grad
        for layer in self.layers[::-1]:
            input_grad = layer.backward(input_grad, self.lr)
            
    @contextmanager
    def no_grad(self):
        self.training = False
        yield
        self.training = True
    
    def train(self, X: ndarray, y: ndarray, epochs: int = 1000, batch_size: int = 32) -> ndarray:
        inputs,targets = X,y
        N = inputs.shape[0]
        if len(targets.shape) == 1:
            targets = targets[:,np.newaxis]
        loss_per_epochs = np.zeros(epochs)
        for e in range(epochs):
            batch_idx = np.random.permutation(N)
            batch_iter = (N // batch_size) + (1 if N % batch_size else 0)
            cumm_loss = 0
            for b in range(batch_iter):
                batch = batch_idx[b * batch_size: (b+1) * batch_size]
                batch_inputs, batch_targets = inputs[batch,:], targets[batch,:]
                batch_output = self.forward(batch_inputs)
                cumm_loss += self.loss.base(batch_targets, batch_output)
                gradient = self.loss.grad(batch_targets, batch_output)
                self.optimizer.step(gradient)
            loss_per_epochs[e] = cumm_loss / batch_iter
        return loss_per_epochs
        
    def __call__(self, _input: ndarray) -> ndarray:
        return self.forward(_input)

                
                
