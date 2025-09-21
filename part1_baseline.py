import tensorflow as tf 

from tensorflow import keras 

import numpy as np 

import time 

import os 

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('part1_baseline.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)
 

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

    logger.info(model.summary())

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

    logger.info(f"Loaded and preprocessed CIFAR-10 dataset")

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
    model_checkpoint = keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, reduce_lr, model_checkpoint]

    # Train for maximum 50 epochs 
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=50,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks)
    
    # Model size analysis (parameters, memory footprint)
    total_params = model.count_params()
    weights_memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter
    
    logger.info(f"\n📊 Model Size Analysis:")
    logger.info(f"   Total Parameters: {total_params:,}")
    logger.info(f"   Model Weights: {weights_memory_mb:.2f} MB")
    logger.info(f"   Training Memory (est.): {weights_memory_mb * 3:.2f} MB")
    logger.info(f"   Inference Memory (est.): {weights_memory_mb * 1.1:.2f} MB")
    
    # Baseline performance metrics (accuracy, inference time, model size)
    # Calculate test accuracy
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    # Measure inference time
    start_time = time.time()
    _ = model.predict(x_test[:100], verbose=0)  # Predict on 100 samples
    inference_time = time.time() - start_time
    avg_inference_time = inference_time / 100 * 1000  # Convert to milliseconds per sample
    
    logger.info(f"\n⚡ Performance Metrics:")
    logger.info(f"   Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"   Test Loss: {test_loss:.4f}")
    logger.info(f"   Avg Inference Time: {avg_inference_time:.2f} ms/sample")
    
    # Create metrics dictionary
    metrics = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'total_parameters': total_params,
        'model_size_mb': weights_memory_mb,
        'inference_time_ms': avg_inference_time
    }
    
    return (model, history, metrics) 

 

if __name__ == "__main__": 
    
    # Use GPU if available
    if tf.config.list_physical_devices('GPU'):
        tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
        logger.info("🚀 GPU available, using GPU!")
    else:
        logger.info("🚫 No GPU available, falling back to CPU")

    # Load data 
    
    x_train, y_train, x_test, y_test = load_and_preprocess_data() 

     

    # Create and train baseline model 

    model = create_baseline_model() 

    model, history, metrics = train_baseline_model(model, x_train, y_train, x_test, y_test) 

     

    # Save baseline model 

    model.save('baseline_model.keras') 