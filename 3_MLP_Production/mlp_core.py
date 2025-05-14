import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
import pickle
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional

class BatchNorm:
    def __init__(self, size: int, eps: float = 1e-5, momentum: float = 0.9):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(size)
        self.beta = np.zeros(size)
        self.running_mean = np.zeros(size)
        self.running_var = np.ones(size)
        
        # Cache for backward pass
        self.cache = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            x_hat = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            self.cache = (x, batch_mean, batch_var, x_hat)
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        return self.gamma * x_hat + self.beta
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, mu, var, x_hat = self.cache
        m = x.shape[0]
        
        dgamma = np.sum(dout * x_hat, axis=0)
        dbeta = np.sum(dout, axis=0)
        
        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (x - mu) * -0.5 * (var + self.eps)**(-1.5), axis=0)
        dmu = np.sum(dx_hat * -1 / np.sqrt(var + self.eps), axis=0) + dvar * np.mean(-2 * (x - mu), axis=0)
        dx = dx_hat / np.sqrt(var + self.eps) + dvar * 2 * (x - mu) / m + dmu / m
        
        return dx

class EnhancedMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_size: int = 8, learning_rate: float = 0.01, 
                 lambda_reg: float = 0.01, keep_prob: float = 0.8, 
                 use_batchnorm: bool = False, optimizer: str = 'adam'):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.keep_prob = keep_prob
        self.use_batchnorm = use_batchnorm
        self.optimizer = optimizer
        self.is_fitted_ = False
    
    def _init_params(self, input_size: int):
        # He initialization
        self.W1 = np.random.randn(input_size, self.hidden_size) * np.sqrt(2./input_size)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, 1) * np.sqrt(2./self.hidden_size)
        self.b2 = np.zeros(1)
        
        if self.use_batchnorm:
            self.bn = BatchNorm(self.hidden_size)
        
        if self.optimizer == 'adam':
            self._init_adam()
    
    def _init_adam(self):
        self.m = {
            'W1': np.zeros_like(self.W1),
            'b1': np.zeros_like(self.b1),
            'W2': np.zeros_like(self.W2),
            'b2': np.zeros_like(self.b2)
        }
        self.v = {
            'W1': np.zeros_like(self.W1),
            'b1': np.zeros_like(self.b1),
            'W2': np.zeros_like(self.W2),
            'b2': np.zeros_like(self.b2)
        }
        self.t = 0
    
    def _relu(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)
    
    def _sigmoid(self, Z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-Z))
    
    def _compute_loss(self, Y: np.ndarray, A2: np.ndarray) -> float:
        m = Y.shape[0]
        cross_entropy = -np.mean(Y * np.log(A2) + (1-Y) * np.log(1-A2))
        l2_reg = (np.sum(self.W1**2) + np.sum(self.W2**2)) * self.lambda_reg / (2 * m)
        return cross_entropy + l2_reg
    
    def _forward(self, X: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict]:
        # Hidden layer
        Z1 = np.dot(X, self.W1) + self.b1
        if self.use_batchnorm:
            A1 = self._relu(self.bn.forward(Z1, training))
        else:
            A1 = self._relu(Z1)
        
        # Apply dropout if training
        if training and hasattr(self, 'keep_prob'):
            D1 = (np.random.rand(*A1.shape) < self.keep_prob).astype(int)
            A1 *= D1 / self.keep_prob
        else:
            D1 = None
        
        # Output layer
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._sigmoid(Z2)
        
        cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'D1': D1}
        return A2, cache
    
    def _backward(self, X: np.ndarray, Y: np.ndarray, cache: Dict) -> Dict:
        m = X.shape[0]
        A2 = cache['A2']
        
        # Output layer gradients
        dZ2 = A2 - Y
        dW2 = np.dot(cache['A1'].T, dZ2) / m + (self.lambda_reg * self.W2) / m
        db2 = np.sum(dZ2, axis=0) / m
        
        # Backprop through dropout
        dA1 = np.dot(dZ2, self.W2.T)
        if cache['D1'] is not None:
            dA1 *= cache['D1'] / self.keep_prob
        
        # Backprop through batchnorm
        if self.use_batchnorm:
            dZ1 = self.bn.backward(dA1)
        else:
            dZ1 = dA1 * (cache['Z1'] > 0).astype(int)
        
        dW1 = np.dot(X.T, dZ1) / m + (self.lambda_reg * self.W1) / m
        db1 = np.sum(dZ1, axis=0) / m
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def _update_params(self, grads: Dict, learning_rate: float):
        if self.optimizer == 'adam':
            self.t += 1
            for param in ['W1', 'b1', 'W2', 'b2']:
                self.m[param] = 0.9 * self.m[param] + (1 - 0.9) * grads[f'd{param}']
                self.v[param] = 0.999 * self.v[param] + (1 - 0.999) * (grads[f'd{param}']**2)
                
                m_hat = self.m[param] / (1 - 0.9**self.t)
                v_hat = self.v[param] / (1 - 0.999**self.t)
                
                getattr(self, param) -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        else:
            # Vanilla SGD
            self.W1 -= learning_rate * grads['dW1']
            self.b1 -= learning_rate * grads['db1']
            self.W2 -= learning_rate * grads['dW2']
            self.b2 -= learning_rate * grads['db2']
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
            validation_data: Optional[Tuple] = None, verbose: bool = True) -> Dict:
        X, y = check_X_y(X, y)
        y = y.reshape(-1, 1)
        self._init_params(X.shape[1])
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(1, epochs+1):
            # Forward pass
            A2, cache = self._forward(X, training=True)
            
            # Compute loss
            train_loss = self._compute_loss(y, A2)
            history['train_loss'].append(train_loss)
            train_acc = self._compute_accuracy(X, y)
            history['train_acc'].append(train_acc)
            
            # Backward pass
            grads = self._backward(X, y, cache)
            
            # Update parameters
            self._update_params(grads, self.learning_rate)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss, val_acc = self.evaluate(X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if verbose and epoch % 100 == 0:
                    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")
            
            elif verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2%}")
        
        self.is_fitted_ = True
        return history
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        A2, _ = self._forward(X, training=False)
        return A2
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)
    
    def _compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        y = y.reshape(-1, 1)
        A2, _ = self._forward(X, training=False)
        loss = self._compute_loss(y, A2)
        acc = self._compute_accuracy(X, y)
        return loss, acc
    
    def save(self, directory: str):
        Path(directory).mkdir(exist_ok=True)
        
        # Save model weights
        weights = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }
        with open(f'{directory}/weights.pkl', 'wb') as f:
            pickle.dump(weights, f)
        
        # Save metadata
        metadata = {
            'hidden_size': self.hidden_size,
            'input_size': self.W1.shape[0],
            'training_date': datetime.now().isoformat(),
            'params': {
                'learning_rate': self.learning_rate,
                'lambda_reg': self.lambda_reg,
                'keep_prob': self.keep_prob,
                'use_batchnorm': self.use_batchnorm,
                'optimizer': self.optimizer
            }
        }
        with open(f'{directory}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save ONNX version
        self._export_onnx(directory)
    
    def _export_onnx(self, directory: str):
        input_size = self.W1.shape[0]
        
        # Create ONNX model structure
        input_tensor = helper.make_tensor_value_info(
            'input', onnx.TensorProto.FLOAT, [None, input_size])
        output_tensor = helper.make_tensor_value_info(
            'output', onnx.TensorProto.FLOAT, [None, 1])
        
        # Prepare nodes
        nodes = [
            helper.make_node('MatMul', ['input', 'W1'], ['matmul1']),
            helper.make_node('Add', ['matmul1', 'b1'], ['add1']),
            helper.make_node('Relu', ['add1'], ['relu1']),
            helper.make_node('MatMul', ['relu1', 'W2'], ['matmul2']),
            helper.make_node('Add', ['matmul2', 'b2'], ['add2']),
            helper.make_node('Sigmoid', ['add2'], ['output'])
        ]
        
        # Create the model
        graph = helper.make_graph(
            nodes,
            'mlp_model',
            [input_tensor],
            [output_tensor],
            initializer=[
                numpy_helper.from_array(self.W1, 'W1'),
                numpy_helper.from_array(self.b1, 'b1'),
                numpy_helper.from_array(self.W2, 'W2'),
                numpy_helper.from_array(self.b2, 'b2')
            ]
        )
        
        model = helper.make_model(graph)
        onnx.save(model, f'{directory}/model.onnx')
    
    @classmethod
    def load(cls, directory: str):
        with open(f'{directory}/weights.pkl', 'rb') as f:
            weights = pickle.load(f)
        
        with open(f'{directory}/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Reconstruct model
        model = cls(
            hidden_size=metadata['hidden_size'],
            **metadata['params']
        )
        
        model.W1 = weights['W1']
        model.b1 = weights['b1']
        model.W2 = weights['W2']
        model.b2 = weights['b2']
        model.is_fitted_ = True
        
        if model.use_batchnorm:
            model.bn = BatchNorm(metadata['hidden_size'])
        
        if model.optimizer == 'adam':
            model._init_adam()
        
        return model

class DriftDetector:
    def __init__(self, reference_data: np.ndarray, alpha: float = 0.05):
        self.reference_data = reference_data
        self.alpha = alpha
        self.reference_mean = np.mean(reference_data, axis=0)
        self.reference_std = np.std(reference_data, axis=0)
    
    def check_drift(self, new_data: np.ndarray) -> Dict:
        results = {}
        
        # Kolmogorov-Smirnov test for each feature
        ks_results = []
        for i in range(new_data.shape[1]):
            stat, p = stats.ks_2samp(self.reference_data[:,i], new_data[:,i])
            ks_results.append({
                'statistic': stat,
                'p_value': p,
                'drift_detected': p < self.alpha
            })
        
        # Population stability index
        psi = self._calculate_psi(new_data)
        
        results['ks_test'] = ks_results
        results['psi'] = psi
        results['overall_drift'] = any(f['drift_detected'] for f in ks_results) or any(v > 0.25 for v in psi.values())
        
        return results
    
    def _calculate_psi(self, new_data: np.ndarray) -> Dict:
        psi_results = {}
        for i in range(new_data.shape[1]):
            # Create 10 bins based on reference distribution
            min_val = min(self.reference_data[:,i].min(), new_data[:,i].min())
            max_val = max(self.reference_data[:,i].max(), new_data[:,i].max())
            bins = np.linspace(min_val, max_val, 11)
            
            # Calculate percentages
            ref_perc, _ = np.histogram(self.reference_data[:,i], bins=bins)
            ref_perc = ref_perc / len(self.reference_data)
            
            new_perc, _ = np.histogram(new_data[:,i], bins=bins)
            new_perc = new_perc / len(new_data)
            
            # Calculate PSI
            psi = np.sum((new_perc - ref_perc) * np.log((new_perc + 1e-6) / (ref_perc + 1e-6)))
            psi_results[f'feature_{i}'] = psi
        
        return psi_results