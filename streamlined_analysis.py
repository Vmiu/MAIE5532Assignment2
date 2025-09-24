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
    
    logger.info(f"   Total Parameters: {total_params:,}")
    logger.info(f"   Trainable Parameters: {trainable_params:,}")
    
    # Get actual file size
    file_size_bytes = os.path.getsize(model_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Calculate theoretical memory - more accurate for mixed precision
    # For mixed precision, we need to consider different dtypes per layer
    total_weight_memory = 0
    for layer in model.layers:
        if hasattr(layer, 'count_params') and layer.count_params() > 0:
            layer_params = layer.count_params()
            layer_dtype = layer.dtype
            if layer_dtype == 'float32':
                dtype_size_bytes = 4
            elif layer_dtype == 'float16':
                dtype_size_bytes = 2
            elif layer_dtype == 'bfloat16':
                dtype_size_bytes = 2
            elif layer_dtype == 'int8':
                dtype_size_bytes = 1
            else:
                dtype_size_bytes = 4  # Default to float32
            
            total_weight_memory += layer_params * dtype_size_bytes
            logger.info(f"   Layer {layer.name}: {layer_params:,} params, dtype={layer_dtype}, policy={layer.dtype_policy}, size={layer_params * dtype_size_bytes / (1024):.2f} KB")
    
    theoretical_memory_mb = total_weight_memory / (1024 * 1024)
    
    # Training memory estimation
    # Use float32 for input data (standard practice)
    input_dtype_size = 4  # float32 for input data
    input_shape = x_test.shape[1:]
    input_memory = batch_size * np.prod(input_shape) * input_dtype_size
    gradients_memory = total_weight_memory  # Same as model weights
    optimizer_memory = total_weight_memory * 2  # Adam optimizer (momentum + variance)
    total_training_memory = total_weight_memory + input_memory + gradients_memory + optimizer_memory
    
    # Inference memory
    inference_memory = total_weight_memory + (np.prod(input_shape) * input_dtype_size)
    
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
    
    # Optimize model for inference
    model.trainable = False
    
    # Convert to TensorFlow function for faster inference
    @tf.function
    def predict_fn(x):
        return model(x, training=False)
    
    # Warm-up - comprehensive GPU warm-up
    warmup_data = tf.constant(x_test[:32], dtype=tf.float32)
    for _ in range(20):
        _ = predict_fn(warmup_data)
    
    # Single sample inference with proper timing
    single_sample_times = []
    single_sample = tf.constant(x_test[:1], dtype=tf.float32)
    
    for _ in range(200):  # More iterations for better statistics
        start_time = time.perf_counter()  # Use high-precision timer
        _ = predict_fn(single_sample)
        end_time = time.perf_counter()
        single_sample_times.append((end_time - start_time) * 1000)
    
    avg_single_time = np.mean(single_sample_times)
    std_single_time = np.std(single_sample_times)
    
    logger.info(f"   Single Sample Time: {avg_single_time:.2f} ± {std_single_time:.2f} ms")
    
    # Check if GPU is being used
    if tf.config.list_physical_devices('GPU'):
        logger.info(f"   GPU Available: Yes")
        logger.info(f"   GPU Memory Growth: {tf.config.experimental.get_memory_growth(tf.config.list_physical_devices('GPU')[0])}")
    else:
        logger.info(f"   GPU Available: No")
    
    results['performance_speed'] = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'single_sample_time_ms': avg_single_time,
        'single_sample_std': std_single_time,
    }
    
    return results
