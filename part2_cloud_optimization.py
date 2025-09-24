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

        Optimize for large batch processing typical in cloud environments. 

         

        Args: 

            target_batch_size: Target batch size for cloud deployment 

             

        Returns: 

            dict: Optimized training configuration 

        """ 

        # TODO: Implement gradient accumulation for large effective batch sizes
        physical_batch_size = 64
        accumulation_steps = target_batch_size // physical_batch_size
        
        part2_logger.info("Gradient Accumulation Configuration:")
        part2_logger.info(f"Physical Batch Size: {physical_batch_size}")
        part2_logger.info(f"Target Batch Size: {target_batch_size}")
        part2_logger.info(f"Accumulation Steps: {accumulation_steps}")

        # TODO: Optimize data pipeline for high throughput
        # TODO: Configure prefetching and parallelism
        def create_optimized_dataset(x, y, batch_size, shuffle_buffer_size=10000):
            """
            Create optimized tf.data pipeline with prefetching and parallelism
            """
            # Convert to tf.data.Dataset
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
            
            # Shuffle with optimal buffer size
            if shuffle_buffer_size:
                dataset = dataset.shuffle(
                    buffer_size=shuffle_buffer_size,
                    reshuffle_each_iteration=True
                )
            
            # Batch the data
            dataset = dataset.batch(batch_size, drop_remainder=True)
            
            # Optimize for performance
            dataset = dataset.map(
                lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # Prefetch for optimal performance
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset
    
        # Create optimized dataset
        train_dataset = create_optimized_dataset(
            x_train, y_train, 
            physical_batch_size,
            shuffle_buffer_size=min(10000, len(x_train))
        )

        # Gradient Accumulation training function
        def gradient_accumulation_training(model, dataset, optimizer, loss_fn, 
                                        accumulation_steps, epochs=10, 
                                        validation_data=None):
            """
            Training function with gradient accumulation
            
            Args:
                model: Keras model to train
                dataset: tf.data.Dataset for training
                optimizer: Keras optimizer
                loss_fn: Loss function
                accumulation_steps: Number of steps to accumulate gradients
                epochs: Number of training epochs
                validation_data: Optional validation dataset
            """
            
            # Metrics
            train_loss_metric = tf.keras.metrics.Mean()
            train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
            
            if validation_data:
                val_loss_metric = tf.keras.metrics.Mean()
                val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
            
            @tf.function
            def accumulate_gradients(x_batch, y_batch):
                """Accumulate gradients for a single batch"""
                with tf.GradientTape() as tape:
                    predictions = model(x_batch, training=True)
                    # Scale loss by accumulation steps to get proper average
                    loss = loss_fn(y_batch, predictions) / accumulation_steps
                
                # Calculate gradients
                gradients = tape.gradient(loss, model.trainable_variables)
                return gradients, loss, predictions
            
            @tf.function
            def apply_accumulated_gradients(accumulated_gradients):
                """Apply accumulated gradients to model"""
                optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
            
            @tf.function
            def validation_step(x_batch, y_batch):
                """Validation step"""
                predictions = model(x_batch, training=False)
                loss = loss_fn(y_batch, predictions)
                return loss, predictions
            
            # Training loop
            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")
                
                # Reset metrics
                train_loss_metric.reset_states()
                train_accuracy_metric.reset_states()
                
                # Initialize accumulated gradients
                accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
                
                step_count = 0
                batch_count = 0
                epoch_start_time = time.time()
                
                for x_batch, y_batch in dataset:
                    batch_count += 1
                    
                    # Accumulate gradients
                    gradients, loss, predictions = accumulate_gradients(x_batch, y_batch)
                    
                    # Add to accumulated gradients
                    for i, grad in enumerate(gradients):
                        accumulated_gradients[i] = accumulated_gradients[i] + grad
                    
                    # Update metrics
                    train_loss_metric.update_state(loss * accumulation_steps)  # Unscale for display
                    train_accuracy_metric.update_state(y_batch, predictions)
                    
                    step_count += 1
                    
                    # Apply gradients when accumulation is complete
                    if step_count % accumulation_steps == 0:
                        apply_accumulated_gradients(accumulated_gradients)
                        
                        # Reset accumulated gradients
                        accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
                        
                        # Print progress
                        if (step_count // accumulation_steps) % 10 == 0:
                            current_loss = train_loss_metric.result()
                            current_acc = train_accuracy_metric.result()
                            print(f"Step {step_count // accumulation_steps} - "
                                f"Loss: {current_loss:.4f} - "
                                f"Accuracy: {current_acc:.4f}")
                
                # Apply any remaining accumulated gradients
                if step_count % accumulation_steps != 0:
                    # Scale remaining gradients properly
                    scale_factor = accumulation_steps / (step_count % accumulation_steps)
                    scaled_gradients = [grad * scale_factor for grad in accumulated_gradients]
                    apply_accumulated_gradients(scaled_gradients)
                
                epoch_time = time.time() - epoch_start_time
                
                # Validation
                if validation_data:
                    val_loss_metric.reset_states()
                    val_accuracy_metric.reset_states()
                    
                    for x_val, y_val in validation_data:
                        val_loss, val_predictions = validation_step(x_val, y_val)
                        val_loss_metric.update_state(val_loss)
                        val_accuracy_metric.update_state(y_val, val_predictions)
                    
                    print(f"Epoch {epoch + 1} - "
                        f"Time: {epoch_time:.2f}s - "
                        f"Loss: {train_loss_metric.result():.4f} - "
                        f"Accuracy: {train_accuracy_metric.result():.4f} - "
                        f"Val Loss: {val_loss_metric.result():.4f} - "
                        f"Val Accuracy: {val_accuracy_metric.result():.4f}")
                else:
                    print(f"Epoch {epoch + 1} - "
                        f"Time: {epoch_time:.2f}s - "
                        f"Loss: {train_loss_metric.result():.4f} - "
                        f"Accuracy: {train_accuracy_metric.result():.4f}")
            
                return model
        return {
            "physical_batch_size": physical_batch_size,
            "accumulation_steps": accumulation_steps,
            "train_dataset": train_dataset,
            "gradient_accumulation_training": gradient_accumulation_training
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

    # # Benchmark mixed precision 
    # MixedPrecisionModel = optimizer.implement_mixed_precision()

    # # TODO: Measure training time, memory usage, accuracy 
    # start_time = time.time()
    # history = MixedPrecisionModel.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test), verbose=0)
    # end_time = time.time()
    # part2_logger.info(f"Training time: {end_time - start_time} seconds")
    
    # # # Save the model for analysis
    # MixedPrecisionModel.save('part2_cloud_optimization/mixed_precision_model.keras')

    # # # Use streamlined analysis for comprehensive metrics
    # mixed_precision_metrics = streamlined_model_analysis(
    #     MixedPrecisionModel, x_test, y_test, 64, 'part2_cloud_optimization/mixed_precision_model.keras', part2_logger
    # )
    
    # metrics['mixed_precision'] = mixed_precision_metrics
    
    ############################################################
    ############################################################
    
    # # Benchmark model parallelism 
    # OneDeviceModel, tf_strategy = optimizer.implement_model_parallelism(strategy='onedevice')
    # DistributedModel, tf_strategy = optimizer.implement_model_parallelism(strategy='mirrored')
    # # TODO: Measure scaling efficiency across multiple GPUs 
    # OneDeviceTime_start = time.time()
    # OneDeviceModel.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test), verbose=0)
    # OneDevice_training_time = time.time() - OneDeviceTime_start
    # DistributedTime_start = time.time()
    # DistributedModel.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test), verbose=0)
    # Distributed_training_time = time.time() - DistributedTime_start
    # part2_logger.info(f"OneDevice training time: {OneDevice_training_time} seconds")
    # part2_logger.info(f"Distributed training time: {Distributed_training_time} seconds")
    # part2_logger.info(f"scaling efficiency: {OneDevice_training_time / Distributed_training_time}")
    
    # metrics['Model Parallelism'] = {
    #     'OneDevice_training_time': OneDevice_training_time, 
    #     'Distributed_training_time': Distributed_training_time, 
    #     'scaling_efficiency': OneDevice_training_time / Distributed_training_time
    # }

    ############################################################
    ############################################################
    
    # Benchmark batch processing optimizations  
    # optimized_batch_processing = optimizer.optimize_batch_processing(target_batch_size=256, x_train=x_train, y_train=y_train)
    # TODO: Measure throughput at different batch sizes 

    
    ############################################################
    ############################################################
    
    # # Benchmark knowledge distillation 
    # teacher_model, student_model, distillation_training_function = optimizer.implement_knowledge_distillation()
    # distillation_training_function(teacher_model, student_model, (x_train, y_train), (x_test, y_test), batch_size=32, epochs=10)

    # # TODO: Measure final student model performance vs teacher 
    # student_model_metrics = streamlined_model_analysis(
    #     student_model, x_test, y_test, 64, 'part2_cloud_optimization/student_model.keras', part2_logger
    # )
    # teacher_model_metrics = streamlined_model_analysis(
    #     teacher_model, x_test, y_test, 64, 'part2_cloud_optimization/teacher_model.keras', part2_logger
    # )
    # results['Knowledge Distillation'] = {
    #     'student_model_metrics': student_model_metrics,
    #     'teacher_model_metrics': teacher_model_metrics
    # }

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