import tensorflow as tf
import numpy as np
import time
import os
import logging

def streamlined_model_analysis(model, x_test, y_test, batch_size, model_path, logger):
    """
    Streamlined model analysis focusing on key metrics.
    
    Args:
        model: Trained TensorFlow model
        x_test: Test input data
        y_test: Test labels
        batch_size: Batch size for inference
        model_path: Path to the model
    
    Returns:
        dict: Analysis results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"🔍 STREAMLINED ANALYSIS: {model_path}")
    logger.info(f"{'='*60}")
    
    results = {}
    
    # 1. ARCHITECTURE & MEMORY ANALYSIS (Merged)
    logger.info(f"\n🏗️  ARCHITECTURE & MEMORY ANALYSIS")
    logger.info(f"-" * 50)
    
    # Architecture info
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    logger.info(f"   Total Parameters: {total_params:,}")
    logger.info(f"   Trainable Parameters: {trainable_params:,}")
    
    # Get actual file size
    file_size_bytes = os.path.getsize(model_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Calculate theoretical memory (float32)
    theoretical_memory_mb = total_params * 4 / (1024 * 1024)
    
    # Training memory estimation
    logger.info(f"   Assume dtype = float32, optimizer = Adam")
    input_shape = x_test.shape[1:]
    input_memory = batch_size * np.prod(input_shape) * 4  # 4 bytes per float32
    gradients_memory = total_params * 4
    optimizer_memory = total_params * 4 * 2  # Adam optimizer
    total_training_memory = theoretical_memory_mb * 1024 * 1024 + input_memory + gradients_memory + optimizer_memory
    
    # Inference memory
    inference_memory = theoretical_memory_mb * 1024 * 1024 + (batch_size * np.prod(input_shape) * 4)
    
    logger.info(f"   Model File Size: {file_size_mb:.2f} MB")
    logger.info(f"   Theoretical Memory: {theoretical_memory_mb:.2f} MB")
    logger.info(f"   Training Memory: {total_training_memory/(1024*1024):.2f} MB")
    logger.info(f"   Inference Memory: {inference_memory/(1024*1024):.2f} MB")  
    logger.info(f"   Memory footprint: {(total_training_memory + inference_memory)/(1024*1024):.2f} MB")  
    
    results['architecture_memory'] = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'file_size_mb': file_size_mb,
        'training_memory_mb': total_training_memory/(1024*1024),
        'inference_memory_mb': inference_memory/(1024*1024),
        'memory_footprint_mb': (total_training_memory + inference_memory)/(1024*1024)
    }
    
    # 2. PERFORMANCE & SPEED ANALYSIS
    logger.info(f"\n⚡ PERFORMANCE & SPEED ANALYSIS")
    logger.info(f"-" * 50)
    
    # Performance metrics
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    logger.info(f"   Test Accuracy:{test_accuracy*100:.2f}%")
    logger.info(f"   Test Loss: {test_loss:.4f}") 
    
    # Warm-up
    _ = model.predict(x_test[:10], verbose=0)
    
    # Single sample inference
    single_sample_times = []
    for _ in range(50):
        start_time = time.time()
        _ = model.predict(x_test[:1], verbose=0)
        single_sample_times.append((time.time() - start_time) * 1000)
    
    avg_single_time = np.mean(single_sample_times)
    std_single_time = np.std(single_sample_times)
    
    logger.info(f"   Single Sample Time: {avg_single_time:.2f} ± {std_single_time:.2f} ms")
    
    results['performance_speed'] = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'single_sample_time_ms': avg_single_time,
        'single_sample_std': std_single_time,
    }
    
    return results
