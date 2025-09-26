import tensorflow as tf 

from tensorflow import keras
from tensorflow.keras import mixed_precision 

import time

import os
# Create directory if it doesn't exist
os.makedirs('part2_cloud_optimization', exist_ok=True)

from logger_utils import get_simple_logger
from part1_baseline import load_and_preprocess_data
from streamlined_analysis import streamlined_model_analysis
class CloudOptimizer: 

    def __init__(self, baseline_model_path): 

        self.baseline_model = keras.models.load_model(baseline_model_path)
         

    def implement_mixed_precision(self): 

        """ 

        Implement mixed precision training/inference for cloud deployment. 

         

        Returns: 

            tf.keras.Model: Model optimized with mixed precision 

        """ 

        # TODO: Enable mixed precision policy 
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

        # TODO: Modify model for mixed precision compatibility
        # Clone the baseline model without using its own dtype_policy
        # Get the model configuration and weights
        model_config = self.baseline_model.get_config()
        model_weights = self.baseline_model.get_weights()
        
        # Create a new model from config
        mixed_precision_model = keras.Sequential.from_config(model_config)
        mixed_precision_model.set_weights(model_weights)
        
        # Explicitly set dtype policy for each layer to override original policies
        for layer in mixed_precision_model.layers:
            if hasattr(layer, 'dtype_policy'):
                # Set mixed precision for all layers except the output layer
                if layer == mixed_precision_model.layers[-1]:
                    # Output layer should use float32 for numerical stability
                    layer.dtype_policy = 'float32'
                else:
                    # All other layers use mixed precision
                    layer.dtype_policy = 'mixed_float16'
        
        # TODO: Handle loss scaling appropriately
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        mixed_precision_model.compile(
            optimizer=optimizer, 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

        return mixed_precision_model

     

    def implement_model_parallelism(self, strategy='mirrored'): 

        """ 

        Implement distributed training strategy for multi-GPU cloud deployment. 

         

        Args: 

            strategy: 'mirrored', 'multi_worker_mirrored', or 'parameter_server' 

             

        Returns: 

            tuple: (distributed_model, training_strategy) 

        """ 

        # TODO: Implement distributed training strategy 
        if strategy == 'mirrored':
            tf_strategy = tf.distribute.MirroredStrategy()
        elif strategy == 'multi_worker_mirrored':
            tf_strategy = tf.distribute.MultiWorkerMirroredStrategy()
        elif strategy == 'parameter_server':
            tf_strategy = tf.distribute.ParameterServerStrategy()
        elif strategy == 'onedevice':
            tf_strategy = tf.distribute.OneDeviceStrategy(device='/GPU:0')
        else:
            part2_logger.error(f"Invalid strategy: {strategy}")
            raise ValueError(f"Invalid strategy: {strategy}")

        # TODO: Adapt model for chosen parallelism approach 
        with tf_strategy.scope():
            model_config = self.baseline_model.get_config()
            model_weights = self.baseline_model.get_weights()
            distributed_model = keras.Sequential.from_config(model_config)
            distributed_model.set_weights(model_weights)

        # TODO: Configure gradient synchronization 
        distributed_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        part2_logger.info(f"Distributed model compiled with {tf_strategy.__class__.__name__}")

        return (distributed_model, tf_strategy)

     

    def optimize_batch_processing(self, x_train, y_train, target_batch_size=256): 

        """ 

        Optimize for large batch processing typical in cloud environments using gradient accumulation. 

         

        Args: 

            x_train: Training data features
            y_train: Training data labels
            target_batch_size: Target effective batch size for cloud deployment 

             

        Returns: 

            dict: Optimized training configuration

        """ 

        # Implement gradient accumulation for large effective batch sizes
        physical_batch_size = 64
        accumulation_steps = target_batch_size // physical_batch_size
        
        part2_logger.info(f"Implementing gradient accumulation:")
        part2_logger.info(f"  Target batch size: {target_batch_size}")
        part2_logger.info(f"  Physical batch size: {physical_batch_size}")
        part2_logger.info(f"  Accumulation steps: {accumulation_steps}")
        
        # Create a copy of the baseline model for gradient accumulation training
        model_config = self.baseline_model.get_config()
        model_weights = self.baseline_model.get_weights()
        ga_model = keras.Sequential.from_config(model_config)
        ga_model.set_weights(model_weights)
        
        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        ga_model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Optimize data pipeline for high throughput
        # Create tf.data.Dataset with prefetching and parallelism
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.batch(physical_batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        train_dataset = train_dataset.cache()
        
        # Training step that handles both normal and gradient accumulation
        def training_step(model, optimizer, x_batch, y_batch, accumulation_steps):
            """Training step that handles both normal training and gradient accumulation"""
            
            try:
                if accumulation_steps == 1:
                    # Normal training - no accumulation needed
                    with tf.GradientTape() as tape:
                        predictions = model(x_batch, training=True)
                        loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
                        loss = tf.reduce_mean(loss)
                    
                    gradients = tape.gradient(loss, model.trainable_variables)
                    
                    # Filter out None gradients
                    valid_gradients = []
                    valid_variables = []
                    if gradients is not None:
                        for grad, var in zip(gradients, model.trainable_variables):
                            if grad is not None:
                                valid_gradients.append(grad)
                                valid_variables.append(var)
                    
                    if valid_gradients:
                        optimizer.apply_gradients(zip(valid_gradients, valid_variables))
                    
                    return loss
                
                else:
                    # Gradient accumulation for larger batch sizes
                    total_loss = 0.0
                    accumulated_gradients = None
                    
                    for i in range(accumulation_steps):
                        start_idx = i * physical_batch_size
                        end_idx = min((i + 1) * physical_batch_size, len(x_batch))
                        
                        if start_idx >= len(x_batch):
                            break
                            
                        x_sub_batch = x_batch[start_idx:end_idx]
                        y_sub_batch = y_batch[start_idx:end_idx]
                        
                        with tf.GradientTape() as tape:
                            predictions = model(x_sub_batch, training=True)
                            loss = tf.keras.losses.sparse_categorical_crossentropy(y_sub_batch, predictions)
                            loss = tf.reduce_mean(loss)
                            scaled_loss = loss / accumulation_steps
                        
                        gradients = tape.gradient(scaled_loss, model.trainable_variables)
                        
                        if gradients is not None:
                            if accumulated_gradients is None:
                                accumulated_gradients = gradients
                            else:
                                accumulated_gradients = [acc_grad + grad for acc_grad, grad in zip(accumulated_gradients, gradients)]
                        
                        total_loss += loss
                    
                    # Apply accumulated gradients
                    valid_gradients = []
                    valid_variables = []
                    if accumulated_gradients is not None:
                        for grad, var in zip(accumulated_gradients, model.trainable_variables):
                            if grad is not None:
                                valid_gradients.append(grad)
                                valid_variables.append(var)
                    
                    if valid_gradients:
                        optimizer.apply_gradients(zip(valid_gradients, valid_variables))
                    
                    return total_loss / accumulation_steps
                    
            except Exception as e:
                part2_logger.error(f"Training step failed: {e}")
                raise e

        return {
            "Physical Batch Size": physical_batch_size,
            "Target Batch Size": target_batch_size,
            "Accumulation Steps": accumulation_steps,
            "Training Step Function": training_step,
            "Optimized Model": ga_model,
            "Optimizer": optimizer
        }
     

    def implement_knowledge_distillation(self): 

        """ 

        Create a larger teacher model and distill knowledge to student model. 

         

        Returns: 

            tuple: (teacher_model, student_model, distillation_training_function) 

        """ 

        # TODO: Create larger teacher model (2x parameters)
        # Create teacher model (same as baseline model)
        teacher_model_config = self.baseline_model.get_config()
        teacher_model = keras.Sequential.from_config(teacher_model_config)
        teacher_model.set_weights(self.baseline_model.get_weights())

        # Create a simpler student model
        # This avoids shape mismatch issues with BatchNormalization layers
        student_model = keras.Sequential([
            keras.layers.Input(shape=(32, 32, 3)),
            
            # Block 1: Reduced filters (16 instead of 32)
            keras.layers.Conv2D(16, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(16, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPooling2D((2, 2)),

            # Block 2: Reduced filters (32 instead of 64)
            keras.layers.Conv2D(32, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(32, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPooling2D((2, 2)),

            # Block 3: Reduced filters (64 instead of 128)
            keras.layers.Conv2D(64, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(64, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPooling2D((2, 2)),
            
            # Classifier: Reduced units (128 instead of 256)
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(10, activation='softmax'),
        ])
        
        # Compile the student model
        student_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # TODO: Implement knowledge distillation loss
        def distillation_loss(y_true, y_student, y_teacher, temperature=3.0, alpha=0.7):
            """
            Compute knowledge distillation loss
            
            Args:
                y_true: True labels
                y_student: Student model predictions
                y_teacher: Teacher model predictions (soft targets)
                temperature: Temperature for softening distributions
                alpha: Weight for distillation loss vs hard target loss
            """
            # Soften the teacher and student predictions
            y_teacher_soft = tf.nn.softmax(y_teacher / temperature)
            y_student_soft = tf.nn.softmax(y_student / temperature)
            
            # Distillation loss (KL divergence between soft predictions)
            distill_loss = keras.losses.KLDivergence()(y_teacher_soft, y_student_soft)
            distill_loss *= (temperature ** 2)  # Scale by temperature squared
            
            # Hard target loss (standard classification loss)
            hard_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_student)
            
            # Combined loss
            return alpha * distill_loss + (1 - alpha) * hard_loss

        # TODO: Set up distillation training loop
        def distillation_training_function(teacher_model, student_model, train_data, val_data, batch_size=32, epochs=10, 
                                        temperature=3.0, alpha=0.7, learning_rate=0.001):
            """
            Train student model using knowledge distillation
            
            Args:
                train_data: Training dataset (x, y)
                val_data: Validation dataset (optional)
                epochs: Number of training epochs
                temperature: Temperature for knowledge distillation
                alpha: Weight for distillation loss
                learning_rate: Learning rate for optimizer
            """
            
            part2_logger.info("Starting knowledge distillation training...")
            
            # Set up optimizer for student model
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            
            # Training metrics
            train_loss_metric = keras.metrics.Mean()
            train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
            val_loss_metric = keras.metrics.Mean()
            val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
            
            @tf.function
            def train_step(x_batch, y_batch):
                with tf.GradientTape() as tape:
                    # Get teacher predictions (no gradient computation needed)
                    teacher_logits = teacher_model(x_batch, training=False)
                    
                    # Get student predictions
                    student_logits = student_model(x_batch, training=True)
                    
                    # Compute distillation loss
                    loss = distillation_loss(y_batch, student_logits, teacher_logits, 
                                        temperature, alpha)
                
                # Compute gradients and update student model
                gradients = tape.gradient(loss, student_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
                
                # Update metrics
                train_loss_metric.update_state(loss)
                train_acc_metric.update_state(y_batch, student_logits)
                
                return loss
            
            @tf.function
            def val_step(x_batch, y_batch):
                teacher_logits = teacher_model(x_batch, training=False)
                student_logits = student_model(x_batch, training=False)
                
                loss = distillation_loss(y_batch, student_logits, teacher_logits, 
                                    temperature, alpha)
                
                val_loss_metric.update_state(loss)
                val_acc_metric.update_state(y_batch, student_logits)
                
                return loss
            
            # Training loop
            x_train, y_train = train_data
            x_val, y_val = val_data
            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")
                
                # Reset metrics
                train_loss_metric.reset_state()
                train_acc_metric.reset_state()
                
                # Training
                num_batches = len(x_train) // batch_size  # Assuming batch size of 32
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(x_train))
                    
                    x_batch = x_train[start_idx:end_idx]
                    y_batch = y_train[start_idx:end_idx]
                    
                    train_step(x_batch, y_batch)
                    
                    if batch_idx % 100 == 0:
                        part2_logger.info(f"Batch {batch_idx}/{num_batches} - "
                            f"Loss: {train_loss_metric.result():.4f} - "
                            f"Acc: {train_acc_metric.result():.4f}")
                
                # Validation
                val_loss_metric.reset_state()
                val_acc_metric.reset_state()
                
                val_num_batches = len(x_val) // batch_size
                
                for batch_idx in range(val_num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(x_val))

                    x_batch = x_val[start_idx:end_idx]
                    y_batch = y_val[start_idx:end_idx]
                    
                    val_step(x_batch, y_batch)
                
                part2_logger.info(f"Epoch {epoch + 1} - "
                    f"Train Loss: {train_loss_metric.result():.4f} - "
                    f"Train Acc: {train_acc_metric.result():.4f} - "
                    f"Val Loss: {val_loss_metric.result():.4f} - "
                    f"Val Acc: {val_acc_metric.result():.4f}")
            
            part2_logger.info("Knowledge distillation training completed!")
            return student_model
        

        return (teacher_model, student_model, distillation_training_function)

 

def benchmark_cloud_optimizations(): 

    """ 

    Benchmark different cloud optimization strategies. 

     

    Returns: 

        dict: Performance metrics for each optimization 

    """ 

    optimizer = CloudOptimizer('part1_baseline/baseline_model.keras') 

    results = {} 

    # Load the data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    ############################################################
    ############################################################

    # Benchmark mixed precision 
    MixedPrecisionModel = optimizer.implement_mixed_precision()

    # TODO: Measure training time, memory usage, accuracy 
    start_time = time.time()
    history = MixedPrecisionModel.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test), verbose=0)
    end_time = time.time()
    part2_logger.info(f"Training time: {end_time - start_time} seconds")
    
    # # Save the model for analysis
    MixedPrecisionModel.save('part2_cloud_optimization/mixed_precision_model.keras')

    # # Use streamlined analysis for comprehensive metrics
    mixed_precision_metrics = streamlined_model_analysis(
        MixedPrecisionModel, x_test, y_test, 64, 'part2_cloud_optimization/mixed_precision_model.keras', part2_logger
    )
    
    # metrics['mixed_precision'] = mixed_precision_metrics
    
    ############################################################
    ############################################################
    
    # Benchmark model parallelism 
    OneDeviceModel, tf_strategy = optimizer.implement_model_parallelism(strategy='onedevice')
    DistributedModel, tf_strategy = optimizer.implement_model_parallelism(strategy='mirrored')
    # TODO: Measure scaling efficiency across multiple GPUs 
    OneDeviceTime_start = time.time()
    OneDeviceModel.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test), verbose=0)
    OneDevice_training_time = time.time() - OneDeviceTime_start
    DistributedTime_start = time.time()
    DistributedModel.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test), verbose=0)
    Distributed_training_time = time.time() - DistributedTime_start
    part2_logger.info(f"OneDevice training time: {OneDevice_training_time} seconds")
    part2_logger.info(f"Distributed training time: {Distributed_training_time} seconds")
    part2_logger.info(f"scaling efficiency: {OneDevice_training_time / Distributed_training_time}")
    
    metrics['Model Parallelism'] = {
        'OneDevice_training_time': OneDevice_training_time, 
        'Distributed_training_time': Distributed_training_time, 
        'scaling_efficiency': OneDevice_training_time / Distributed_training_time
    }

    ############################################################
    ############################################################
    
    # Benchmark batch processing optimizations with gradient accumulation
    part2_logger.info("Starting gradient accumulation batch processing benchmark...")
    batch_processing_config = optimizer.optimize_batch_processing(target_batch_size=256, x_train=x_train, y_train=y_train)
    
    # Extract configuration from the object method
    physical_batch_size = batch_processing_config["Physical Batch Size"]
    training_step_function = batch_processing_config["Training Step Function"]
    
    # Measure throughput for different batch sizes
    batch_sizes_to_test = [64, 128, 256, 512, 1024]
    throughput_results = {}
    
    part2_logger.info("=== GRADIENT ACCUMULATION THROUGHPUT ANALYSIS ===")
    
    for test_batch_size in batch_sizes_to_test:
        part2_logger.info(f"Testing throughput for batch size: {test_batch_size}")
        
        # Calculate accumulation steps
        test_accumulation_steps = test_batch_size // physical_batch_size
        
        if test_accumulation_steps == 0:
            part2_logger.warning(f"Skipping batch size {test_batch_size} - smaller than physical batch size {physical_batch_size}")
            continue
        
        part2_logger.info(f"  Accumulation steps: {test_accumulation_steps}")
        
        # Create fresh model for each test
        test_model_config = optimizer.baseline_model.get_config()
        test_model_weights = optimizer.baseline_model.get_weights()
        test_model = keras.Sequential.from_config(test_model_config)
        test_model.set_weights(test_model_weights)
        
        test_optimizer = keras.optimizers.Adam(learning_rate=0.001)
        test_model.compile(
            optimizer=test_optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Warm up
        warmup_batch = x_train[:test_batch_size]
        warmup_labels = y_train[:test_batch_size]
        training_step_function(test_model, test_optimizer, warmup_batch, warmup_labels, test_accumulation_steps)
        
        # Measure throughput
        start_time = time.time()
        samples_processed = 0
        num_batches = min(10, len(x_train) // test_batch_size)  # Limit to 10 batches
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * test_batch_size
            end_idx = min((batch_idx + 1) * test_batch_size, len(x_train))
            
            x_batch = x_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            training_step_function(test_model, test_optimizer, x_batch, y_batch, test_accumulation_steps)
            samples_processed += len(x_batch)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Calculate throughput metrics
        throughput_samples_per_sec = samples_processed / training_time
        throughput_batches_per_sec = samples_processed / (test_batch_size * training_time)
        
        throughput_results[test_batch_size] = {
            'training_time': training_time,
            'samples_processed': samples_processed,
            'throughput_samples_per_sec': throughput_samples_per_sec,
            'throughput_batches_per_sec': throughput_batches_per_sec,
            'accumulation_steps': test_accumulation_steps,
            'physical_batch_size': physical_batch_size
        }
        
        part2_logger.info(f"  Results: {throughput_samples_per_sec:.2f} samples/sec, {throughput_batches_per_sec:.2f} batches/sec")
    
    # Find optimal batch size based on throughput
    if throughput_results:
        best_batch_size = max(throughput_results.keys(), 
                            key=lambda x: throughput_results[x]['throughput_samples_per_sec'])
        
        part2_logger.info(f"Optimal batch size for throughput: {best_batch_size}")
        part2_logger.info(f"Best throughput: {throughput_results[best_batch_size]['throughput_samples_per_sec']:.2f} samples/sec")
        
        # Combine results
        batch_processing_results = {
            **batch_processing_config,
            "Throughput Results": throughput_results,
            "Optimal Batch Size": best_batch_size,
            "Best Throughput": throughput_results[best_batch_size]['throughput_samples_per_sec']
        }
    else:
        batch_processing_results = batch_processing_config
    
    results['Gradient Accumulation'] = batch_processing_results 

    
    ############################################################
    ############################################################
    
    # Benchmark knowledge distillation 
    teacher_model, student_model, distillation_training_function = optimizer.implement_knowledge_distillation()
    distillation_training_function(teacher_model, student_model, (x_train, y_train), (x_test, y_test), batch_size=32, epochs=10)

    # TODO: Measure final student model performance vs teacher 
    student_model_metrics = streamlined_model_analysis(
        student_model, x_test, y_test, 64, 'part2_cloud_optimization/student_model.keras', part2_logger
    )
    teacher_model_metrics = streamlined_model_analysis(
        teacher_model, x_test, y_test, 64, 'part2_cloud_optimization/teacher_model.keras', part2_logger
    )
    results['Knowledge Distillation'] = {
        'student_model_metrics': student_model_metrics,
        'teacher_model_metrics': teacher_model_metrics
    }

    return results 

 

if __name__ == "__main__": 
    # Create isolated logger for part2
    part2_logger = get_simple_logger("part2_cloud_optimization", "part2_cloud_optimization/part2_terminal.log")
    
    # Use GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        part2_logger.info(f"GPU available: {physical_devices}")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        part2_logger.info(f"No GPU available")

    results = benchmark_cloud_optimizations() 

    print("Cloud Optimization Results:") 

    for optimization, metrics in results.items(): 

        print(f"{optimization}: {metrics}") 