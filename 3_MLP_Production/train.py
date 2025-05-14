import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
import optuna
from mlp_core import EnhancedMLP, DriftDetector
import matplotlib.pyplot as plt
import json

def generate_data(n_samples=1000, n_features=20, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=2,
        n_informative=10,
        n_clusters_per_class=2,
        flip_y=0.1,
        random_state=random_state
    )
    return X.astype(np.float32), y.astype(np.int32)

def objective(trial, X, y):
    params = {
        'hidden_size': trial.suggest_int('hidden_size', 4, 64),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'lambda_reg': trial.suggest_float('lambda_reg', 1e-6, 1e-1, log=True),
        'keep_prob': trial.suggest_float('keep_prob', 0.5, 0.95),
        'use_batchnorm': trial.suggest_categorical('use_batchnorm', [True, False]),
        'optimizer': trial.suggest_categorical('optimizer', ['sgd', 'adam'])
    }
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    val_accs = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = EnhancedMLP(**params)
        model.fit(X_train, y_train, epochs=200, verbose=False)
        _, val_acc = model.evaluate(X_val, y_val)
        val_accs.append(val_acc)
    
    return np.mean(val_accs)

def train_final_model(X_train, y_train, best_params):
    model = EnhancedMLP(**best_params)
    history = model.fit(
        X_train, y_train, 
        epochs=1000,
        verbose=True
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    if 'val_acc' in history:
        plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model

def main():
    # Generate and split data
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Accuracy): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Train final model with best params
    best_params = trial.params
    model = train_final_model(X_train, y_train, best_params)
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.2%}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save model
    model.save('production_model')
    
    # Initialize drift detector
    drift_detector = DriftDetector(X_train)
    drift_results = drift_detector.check_drift(X_test)
    print("\nDrift Detection Results:")
    print(json.dumps(drift_results, indent=2))

if __name__ == "__main__":
    main()