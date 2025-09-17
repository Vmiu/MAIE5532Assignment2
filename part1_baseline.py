import tensorflow as tf 

from tensorflow import keras 

import numpy as np 

import time 

import os 

 

def create_baseline_model(): 

    """ 

    Create a moderately complex CNN for CIFAR-10 classification. 

    This model is intentionally over-parameterized to demonstrate optimization potential. 

     

    Returns: 

        tf.keras.Model: Compiled model ready for training 

    """ 

    model = keras.Sequential([ 

        # TODO: Implement the required layers 

        # Block 1: Conv2D(32, 3x3) -> BatchNorm -> ReLU -> Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) 

        # Block 2: Conv2D(64, 3x3) -> BatchNorm -> ReLU -> Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) 

        # Block 3: Conv2D(128, 3x3) -> BatchNorm -> ReLU -> Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) 

        # Classifier: GlobalAveragePooling2D -> Dropout(0.5) -> Dense(256) -> Dropout(0.3) -> Dense(10) 

    ]) 

     

    model.compile( 

        optimizer='adam', 

        loss='sparse_categorical_crossentropy', 

        metrics=['accuracy'] 

    ) 

     

    return model 

 

def load_and_preprocess_data(): 

    """ 

    Load and preprocess CIFAR-10 dataset. 

     

    Returns: 

        tuple: (x_train, y_train, x_test, y_test) 

    """ 

    # TODO: Load CIFAR-10 dataset 

    # Normalize pixel values to [0, 1] range 

    # Apply data augmentation for training set 

    pass 

 

def train_baseline_model(model, x_train, y_train, x_test, y_test): 

    """ 

    Train the baseline model with early stopping and learning rate scheduling. 

     

    Returns: 

        tuple: (model, training_history, training_metrics) 

    """ 

    # TODO: Implement training with callbacks: 

    # - EarlyStopping (patience=10) 

    # - ReduceLROnPlateau 

    # - ModelCheckpoint 

    # Train for maximum 50 epochs 

    pass 

 

if __name__ == "__main__": 

    # Load data 

    x_train, y_train, x_test, y_test = load_and_preprocess_data() 

     

    # Create and train baseline model 

    model = create_baseline_model() 

    model, history, metrics = train_baseline_model(model, x_train, y_train, x_test, y_test) 

     

    # Save baseline model 

    model.save('baseline_model.keras') 

     

    print(f"Baseline model parameters: {model.count_params():,}") 

    print(f"Baseline test accuracy: {metrics['test_accuracy']:.4f}")