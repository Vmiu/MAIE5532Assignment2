import tensorflow as tf 

from tensorflow import keras 

import numpy as np 

import matplotlib.pyplot as plt

import time 

import os 

from logger_utils import get_simple_logger

# Create directory if it doesn't exist
os.makedirs('part1_baseline', exist_ok=True)
 
from streamlined_analysis import streamlined_model_analysis

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
        keras.layers.Input(shape=(32, 32, 3)),
        keras.layers.Conv2D(32, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(32, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D((2, 2)),

        # Block 2: Conv2D(64, 3x3) -> BatchNorm -> ReLU -> Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) 
        keras.layers.Conv2D(64, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(64, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D((2, 2)),

        # Block 3: Conv2D(128, 3x3) -> BatchNorm -> ReLU -> Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) 
        keras.layers.Conv2D(128, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(128, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Classifier: GlobalAveragePooling2D -> Dropout(0.5) -> Dense(256) -> Dropout(0.3) -> Dense(10) 
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax'),
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
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1] range 
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Apply data augmentation for training set 
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(x_train)
    
    # Generate augmented training data
    augmented_data = []
    augmented_labels = []
    for batch_x, batch_y in datagen.flow(x_train, y_train, batch_size=32, shuffle=False):
        augmented_data.append(batch_x)
        augmented_labels.append(batch_y)
        if len(augmented_data) * 32 >= len(x_train):
            break
    
    # Convert to numpy arrays and concatenate with original data
    augmented_x = np.concatenate(augmented_data, axis=0)[:len(x_train)]
    augmented_y = np.concatenate(augmented_labels, axis=0)[:len(y_train)]
    
    # Replace original training data with augmented data
    x_train = augmented_x
    y_train = augmented_y

    return (x_train, y_train, x_test, y_test) 

 

def train_baseline_model(model, x_train, y_train, x_test, y_test): 

    """ 

    Train the baseline model with early stopping and learning rate scheduling. 

     

    Returns: 

        tuple: (model, training_history, training_metrics) 

    """ 

    # TODO: Implement training with callbacks:
    # - EarlyStopping (patience=10) 
    early_stopping = keras.callbacks.EarlyStopping(patience=10)
    
    # - ReduceLROnPlateau 
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)
    
    # - ModelCheckpoint
    model_checkpoint = keras.callbacks.ModelCheckpoint('part1_baseline/best_model.keras', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, reduce_lr, model_checkpoint]

    # Train for maximum 50 epochs 
    part1_logger.info(f"Training baseline model for 50 epochs")
    time_start = time.time()
    history = model.fit(x_train, y_train,
                        batch_size=64,  # Reduced from 128 to reduce register pressure
                        epochs=50,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks)
    time_end = time.time()
    part1_logger.info(f"🕒 Total time taken to train baseline model: {time_end - time_start:.2f} seconds")
    
    # Plot accuracy and loss
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('part1_baseline/accuracy.png')
    plt.close()
    
    # Streamlined model analysis
    metrics = streamlined_model_analysis(model, x_test, y_test, 64, 'part1_baseline/best_model.keras', part1_logger)
    return (model, history, metrics) 

 

if __name__ == "__main__": 
    part1_logger = get_simple_logger("part1_baseline", "part1_baseline/part1_terminal.log")
    
    # Use GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        part1_logger.info(f"GPU available: {physical_devices}")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        
        # Additional optimizations to reduce CUDA register spilling
        tf.config.optimizer.set_jit(True)  # Enable XLA compilation
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
    else:
        part1_logger.info(f"No GPU available")

    # Load data 
    
    x_train, y_train, x_test, y_test = load_and_preprocess_data() 

     

    # Create and train baseline model 

    model = create_baseline_model() 

    model, history, metrics = train_baseline_model(model, x_train, y_train, x_test, y_test) 

     

    # Save baseline model 

    model.save('part1_baseline/baseline_model.keras') 