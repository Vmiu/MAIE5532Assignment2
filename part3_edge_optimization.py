import tensorflow as tf 

import tensorflow_model_optimization as tfmot 

import matplotlib.pyplot as plt

import numpy as np

import os

import time
from itertools import product

from logger_utils import get_simple_logger

from part1_baseline import load_and_preprocess_data, create_baseline_model

class EdgeOptimizer: 

    def __init__(self, baseline_model_path, x_train, y_train, x_test, y_test, part3_logger): 

        self.baseline_model = tf.keras.models.load_model(baseline_model_path) 
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def count_non_zero_params(self, model):
        """Count the number of non-zero parameters in a model"""
        non_zero_count = 0
        for w in model.trainable_weights:
            non_zero_count += tf.reduce_sum(tf.cast(tf.not_equal(w, 0), tf.int32)).numpy()
        return non_zero_count

    def count_flops(self, model):
        """
        Count the total number of floating point operations (FLOPs) in a model.
        
        Args:
            model: Keras model
            
        Returns:
            int: Total number of FLOPs
        """
        flops = 0
        
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                # Conv2D FLOPs = output_height * output_width * kernel_height * kernel_width * input_channels * output_channels
                output_shape = layer.output_shape[1:]  # Remove batch dimension
                kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
                input_channels = layer.input_shape[-1]
                output_channels = layer.filters
                flops += output_shape[0] * output_shape[1] * kernel_size * input_channels * output_channels
                
            elif isinstance(layer, tf.keras.layers.Dense):
                # Dense FLOPs = input_units * output_units
                input_units = layer.input_shape[-1]
                output_units = layer.units
                flops += input_units * output_units
                
            elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                # DepthwiseConv2D FLOPs = output_height * output_width * kernel_height * kernel_width * input_channels
                output_shape = layer.output_shape[1:]
                kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
                input_channels = layer.input_shape[-1]
                flops += output_shape[0] * output_shape[1] * kernel_size * input_channels
        
        return flops

    def implement_pruning(self, target_sparsity=0.75): 

        """ 

        Implement magnitude-based pruning for edge deployment. 

         

        Args: 

            target_sparsity: Target sparsity level (0.75 = 75% weights pruned) 

             

        Returns: 

            tf.keras.Model: Pruned model 

        """ 

        # TODO: Implement magnitude-based pruning 
        part3_logger.info(f"Implementing magnitude-based pruning with target sparsity: {target_sparsity}")
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=100000
            )
        }

        # TODO: Set up pruning schedule 
        part3_logger.info(f"Setting up pruning schedule")

        # TODO: Fine-tune pruned model 
        part3_logger.info(f"Fine-tuning pruned model")
        pruned_model = create_baseline_model()

        # Set weights from baseline model
        pruned_model.set_weights(self.baseline_model.get_weights())
        
        # Apply pruning
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(pruned_model, **pruning_params)
        pruned_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # TODO: Fine-tune pruned model 
        part3_logger.info(f"Fine-tuning pruned model")
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
        ]
        pruned_model.fit(self.x_train, self.y_train, 
                         epochs=5, batch_size=64, validation_data=(self.x_test, self.y_test), 
                         callbacks=callbacks)

        return pruned_model

     

    def implement_quantization(self): 

        """ 

        Implement post-training quantization for edge deployment. 

         

        Returns: 

            dict: Quantized models with different strategies 

        """ 

        quantized_models = {} 
        
        # TODO: Provide representative dataset for calibration 
        part3_logger.info(f"Providing representative dataset for calibration")
        def representative_data_gen():
            for i in range(100):
                yield [self.x_test[i:i+1].astype(np.float32)]
        dataset = representative_data_gen

        # TODO: Implement dynamic range quantization
        part3_logger.info(f"Implementing dynamic range quantization")
        model = create_baseline_model()
        model.set_weights(self.baseline_model.get_weights())
        converter_dynamic = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_dynamic.optimizations = [tf.lite.Optimize.DEFAULT]
        dynamic_quantized_model = converter_dynamic.convert()
        quantized_models['dynamic_range'] = dynamic_quantized_model
        with open('part3_edge_optimization/dynamic_range.tflite', 'wb') as f:
            f.write(dynamic_quantized_model)
            
        # TODO: Implement full integer quantization 
        part3_logger.info(f"Implementing full integer quantization")
        model = create_baseline_model()
        model.set_weights(self.baseline_model.get_weights())
        converter_full_int = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_full_int.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_full_int.representative_dataset = dataset
        converter_full_int.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_full_int.inference_input_type = tf.int8
        converter_full_int.inference_output_type = tf.int8
        full_int_quantized_model = converter_full_int.convert()
        quantized_models['full_integer'] = full_int_quantized_model
        with open('part3_edge_optimization/full_integer.tflite', 'wb') as f:
            f.write(full_int_quantized_model)

        # TODO: Implement float16 quantization 
        part3_logger.info(f"Implementing float16 quantization")
        model = create_baseline_model()
        model.set_weights(self.baseline_model.get_weights())
        converter_float16 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_float16.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_float16.target_spec.supported_types = [tf.float16]
        float16_quantized_model = converter_float16.convert()
        quantized_models['float16'] = float16_quantized_model
        with open('part3_edge_optimization/float16.tflite', 'wb') as f:
            f.write(float16_quantized_model)

        return quantized_models 

     

    def implement_architecture_optimization(self, architecture={'depth': 3, 'width': 32, 'kernel_size': 3}): 

        """ 

        Optimize model architecture for edge constraints. 

         

        Returns: 

            tf.keras.Model: Architecture-optimized model 

        """ 

        # TODO: Replace standard convolutions with depthwise separable convolutions 
        # TODO: Reduce model depth and width systematically 
        # TODO: Replace global average pooling strategies 
        # TODO: Optimize activation functions for edge inference 
        input_shape = self.x_train.shape[1:]
        num_classes = 10  # Based on baseline model
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))
        
        # Build convolutional blocks based on depth parameter
        current_filters = architecture['width']
        
        for block_idx in range(architecture['depth']):
            # Depthwise separable convolution block
            # Depthwise convolution
            model.add(tf.keras.layers.DepthwiseConv2D(
                kernel_size=architecture['kernel_size'],
                padding='same',
                depth_multiplier=1
            ))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU(max_value=6))
            
            # Pointwise convolution to control output channels
            model.add(tf.keras.layers.Conv2D(
                filters=current_filters,
                kernel_size=1,
                padding='same'
            ))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU(max_value=6))
            
            # Max pooling after each block
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
            
            # Double filters for next block
            current_filters = min(current_filters * 2, 512)  # Cap at 512 to prevent explosion
        
        # Classifier head
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.ReLU(max_value=6))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


    def implement_neural_architecture_search(self): 

        """ 

        Implement simplified NAS for finding optimal edge architecture. 

         

        Returns: 

            tuple: (best_architecture, search_results) 

        """ 

        # Define search space (depth, width, kernel sizes)
        part3_logger.info(f"Defining search space for architecture optimization")
        # Can add more architectures to the search space if needed
        search_space = {
            'depth': [2, 3],
            'width': [16, 32],
            'kernel_size': [3],
        }
        
        # Generate all possible architecture combinations
        architectures = []
        for depth, width, kernel_size in product(search_space['depth'], search_space['width'], search_space['kernel_size']):
            architectures.append({
                'depth': depth,
                'width': width,
                'kernel_size': kernel_size
            })
        
        part3_logger.info(f"Generated {len(architectures)} architecture candidates")
        
        # Evaluate architectures
        search_results = []
        best_architecture = None
        best_score = -np.inf
        
        part3_logger.info(f"Evaluating {len(architectures)} architectures...")
        
        for i, arch in enumerate(architectures):
            part3_logger.info(f"Architecture {i+1}/{len(architectures)}: {arch}")
            
            # Evaluate architecture
            metrics = self._evaluate_architecture(arch)
            
            # Calculate composite score (weighted combination of accuracy and efficiency)
            efficiency_weight = 0.3
            accuracy_weight = 0.7
            
            normalized_accuracy = metrics['accuracy']
            # Normalize efficiency: smaller size and faster inference are better
            max_size = 10.0  # Assume max reasonable size for normalization
            max_time = 100.0  # Assume max reasonable inference time
            
            normalized_size_efficiency = 1.0 - min(metrics['model_size_mb'] / max_size, 1.0)
            normalized_time_efficiency = 1.0 - min(metrics['inference_time_ms'] / max_time, 1.0)
            normalized_efficiency = (normalized_size_efficiency + normalized_time_efficiency) / 2
            
            composite_score = (accuracy_weight * normalized_accuracy + 
                             efficiency_weight * normalized_efficiency)
            
            metrics['composite_score'] = composite_score
            metrics['architecture'] = arch
            search_results.append(metrics)
            
            # Update best architecture
            if composite_score > best_score:
                best_score = composite_score
                best_architecture = arch
                
        # Sort results by composite score
        search_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Find Pareto-optimal architectures
        pareto_optimal = self._find_pareto_optimal(search_results)
        
        part3_logger.info(f"Best architecture found: {best_architecture} with score: {best_score}")
        part3_logger.info(f"Found {len(pareto_optimal)} Pareto-optimal architectures")

        return best_architecture, {
            'all_results': search_results,
            'pareto_optimal': pareto_optimal,
            'best_composite_score': best_score
        }
    
    def _evaluate_architecture(self, architecture):
        """
        Architecture evaluation function based on the search space parameters
        
        Args:
            architecture: Dictionary defining the architecture parameters (depth, width, kernel_size)
            
        Returns:
            dict: Metrics including accuracy, model size, and inference time
        """
        # Build model with given architecture
        model = self.implement_architecture_optimization(architecture)
        
        # Calculate model size
        model_size = self._calculate_model_size(model)
        
        # Train model (with early stopping for efficiency)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        # Use a subset of training data for faster evaluation during NAS
        train_subset_size = min(2000, len(self.x_train))
        indices = np.random.choice(len(self.x_train), train_subset_size, replace=False)
        x_train_subset = self.x_train[indices]
        y_train_subset = self.y_train[indices]
        
        history = model.fit(
            x_train_subset, y_train_subset,
            epochs=15,  # Limited epochs for NAS
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Measure inference time
        inference_time = self._measure_inference_time(model)
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'model_size_mb': model_size,
            'inference_time_ms': inference_time,
            'training_epochs': len(history.history['loss']),
            'parameters': model.count_params()
        }
    

    
    def _calculate_model_size(self, model):
        """Calculate model size in MB"""
        param_count = model.count_params()
        # Assuming 4 bytes per parameter (float32)
        size_mb = (param_count * 4) / (1024 ** 2)
        return size_mb
    
    def _measure_inference_time(self, model):
        """Measure average inference time in milliseconds"""
        # Use a small batch for timing
        sample_batch = self.x_test[:32]  # Larger batch for more stable timing
        
        # Warm up
        for _ in range(3):
            _ = model.predict(sample_batch, verbose=0)
        
        # Measure time over multiple runs
        times = []
        for _ in range(5):
            start_time = time.time()
            _ = model.predict(sample_batch, verbose=0)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate average time per sample in milliseconds
        avg_batch_time = np.mean(times)
        avg_time_ms = (avg_batch_time / len(sample_batch)) * 1000
        return avg_time_ms
    
    def _find_pareto_optimal(self, results):
        """
        Search for Pareto-optimal architectures (accuracy vs efficiency)
        
        Args:
            results: List of evaluation results
            
        Returns:
            list: Pareto-optimal architectures
        """
        pareto_optimal = []
        
        for i, result_i in enumerate(results):
            is_pareto = True
            
            for j, result_j in enumerate(results):
                if i == j:
                    continue
                
                # Check if result_j dominates result_i
                # Better or equal accuracy AND (smaller size AND faster inference)
                accuracy_better_equal = result_j['accuracy'] >= result_i['accuracy']
                size_better = result_j['model_size_mb'] < result_i['model_size_mb']
                time_better = result_j['inference_time_ms'] < result_i['inference_time_ms']
                
                # Strict dominance: at least one metric strictly better, others not worse
                accuracy_strictly_better = result_j['accuracy'] > result_i['accuracy']
                
                # Result j dominates result i if:
                # (Better accuracy AND not worse efficiency) OR (same accuracy AND better efficiency)
                dominates = False
                if accuracy_strictly_better and size_better and time_better:
                    dominates = True
                elif accuracy_better_equal and size_better and time_better and accuracy_strictly_better:
                    dominates = True
                elif result_j['accuracy'] == result_i['accuracy'] and size_better and time_better:
                    dominates = True
                
                if dominates:
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_optimal.append(result_i)
        
        return pareto_optimal

     

    def create_tflite_models(self, models_dict): 

        """ 

        Convert optimized models to TensorFlow Lite format. 

         

        Args: 

            models_dict: Dictionary of optimized Keras models 

             

        Returns: 

            dict: TensorFlow Lite models with metadata 

        """ 

        # TODO: Convert each model to TFLite 
        part3_logger.info(f"Converting models to TFLite")
        tflite_models = {}
        for model_name, model in models_dict.items():
            part3_logger.info(f"Converting {model_name} to TFLite")
            tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
            tflite_models[model_name] = tflite_model

        # TODO: Measure model sizes and inference times 
        part3_logger.info(f"Measuring model sizes and inference times")
        for model_name, model in models_dict.items():
            part3_logger.info(f"Measuring {model_name} model size and inference time")
            model_size = model.count_params()
            
            inference_time = model.evaluate(self.x_test, self.y_test, verbose=0)
            tflite_models[model_name] = {
                'model_size': model_size,
                'inference_time': inference_time
            }

        # TODO: Test accuracy preservation 
        part3_logger.info(f"Testing accuracy preservation")
        
        return tflite_models

 

def benchmark_edge_optimizations(): 

    """ 

    Comprehensive benchmarking of edge optimization strategies. 

     

    Returns: 

        dict: Detailed performance analysis 

    """ 
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    optimizer = EdgeOptimizer('part1_baseline/best_model.keras', x_train, y_train, x_test, y_test, part3_logger) 
    
    results = {} 

    baseline_model = optimizer.baseline_model
    base_acc = baseline_model.evaluate(x_test, y_test, verbose=0)[1]
    results['baseline'] = {
        'accuracy': base_acc,
    }

    #==============================================================

    # Test pruning at different sparsity levels 
    part3_logger.info(f"Testing pruning at different sparsity levels")
    results['pruning'] = {}
    baseline_model = optimizer.baseline_model
    base_acc = baseline_model.evaluate(x_test, y_test, verbose=0)[1]
    
    # Save baseline model and measure size
    baseline_model.save('part3_edge_optimization/baseline_model.keras')
    baseline_size = optimizer.count_non_zero_params(baseline_model)
    results['pruning']['0'] = {
        'accuracy': base_acc,
        'model_size': baseline_size,
    }
    part3_logger.info(f"Baseline model: accuracy={base_acc:.4f}, non-zero params={baseline_size}")
    
    sparsity_levels = [0.25,0.5,0.75]
    for sparsity_level in sparsity_levels:
        part3_logger.info(f"Creating pruned model with sparsity: {sparsity_level}")
        PrunedModel = optimizer.implement_pruning(target_sparsity=sparsity_level)
        acc = PrunedModel.evaluate(x_test, y_test, verbose=0)[1]
        
        # Calculate model size based on non-zero parameters
        model_size = optimizer.count_non_zero_params(PrunedModel)
        
        # Save pruned model
        PrunedModel.save(f'part3_edge_optimization/pruned_model_sparsity_{sparsity_level}.keras')
        
        results['pruning'][str(sparsity_level)] = {
            'accuracy': acc,
            'model_size': model_size,
        }
        part3_logger.info(f"Pruned model (sparsity {sparsity_level}): accuracy={acc:.4f}, non-zero params={model_size}")
    
    # Measure accuracy vs model size trade-offs 
    sparsity_levels = list(results['pruning'].keys())
    pruning_accuracies = [results['pruning'][level]['accuracy'] for level in sparsity_levels]
    pruning_sizes = [results['pruning'][level]['model_size'] for level in sparsity_levels]
    
    # Convert sparsity levels to float for plotting
    sparsity_levels_float = [float(level) for level in sparsity_levels]
    
    # Create accuracy vs sparsity plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sparsity_levels_float, pruning_accuracies, marker='o')
    plt.xlabel('Sparsity Level')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Sparsity Level')
    plt.grid(True)
    
    # Create model size vs sparsity plot
    plt.subplot(1, 2, 2)
    plt.plot(sparsity_levels_float, pruning_sizes, marker='s', color='red')
    plt.xlabel('Sparsity Level')
    plt.ylabel('Non-zero Parameters')
    plt.title('Non-zero Parameters vs Sparsity Level')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('part3_edge_optimization/pruning_analysis.png')
    plt.close()
    
    #==============================================================

    # Test different quantization strategies 
    part3_logger.info(f"Testing different quantization strategies")
    results['quantization'] = {}
    quantized_models = optimizer.implement_quantization()
    for model_name, model_bytes in quantized_models.items():
        # Evaluate TensorFlow Lite model using interpreter
        interpreter = tf.lite.Interpreter(model_content=model_bytes)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare test data
        correct_predictions = 0
        total_samples = len(x_test)

        
        for i in range(total_samples):
            # Prepare input data based on model's expected input type
            if input_details[0]['dtype'] == np.int8:
                input_data = (x_test[i:i+1] * 255).astype(np.int8)
            else:
                input_data = x_test[i:i+1].astype(np.float32)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output_data[0])
            
            if predicted_class == y_test[i]:
                correct_predictions += 1
        
        acc = correct_predictions / total_samples
        model_size = os.path.getsize('part3_edge_optimization/' + model_name + '.tflite')
        results['quantization'][model_name] = {
            'accuracy': acc,
            'model_size': model_size,
        }

    # TODO: Compare INT8, float16, dynamic range quantization 
    quantization_strategies = list(results['quantization'].keys())
    quantization_accuracies = [results['quantization'][strategy]['accuracy'] for strategy in quantization_strategies]
    
    plt.bar(quantization_strategies, quantization_accuracies)
    plt.xlabel('Quantization Strategy')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Quantization Strategy')
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.tight_layout()
    plt.savefig('part3_edge_optimization/quantization_accuracy.png')
    plt.close()

    #==============================================================

    # Test architecture optimizations 
    part3_logger.info(f"Testing architecture optimizations")
    results['architecture'] = {}
    best_architecture, search_results = optimizer.implement_neural_architecture_search()
    architecture_model = optimizer.implement_architecture_optimization(best_architecture)

    # TODO: Measure latency, energy usage estimates, model size 
    # Latency
    num_flops = optimizer.count_flops(architecture_model)
    device_flops = 10e6
    latency = num_flops / device_flops
    # Energy usage
    energy_per_flops = 1e-9
    total_energy = num_flops * energy_per_flops
    # Model size
    model_size = optimizer.count_non_zero_params(architecture_model)
    
    results['architecture'] = {
        'latency_ms': latency,
        'energy_uj': total_energy,
        'model_size': model_size,
    }
     
    #==============================================================

    return results 

 

if __name__ == "__main__": 
    part3_logger = get_simple_logger('part3_edge_optimization', 'part3_edge_optimization/part3_terminal.log')

    # Use GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        part3_logger.info(f"GPU available: {physical_devices}")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        part3_logger.info(f"No GPU available")
        
    results = benchmark_edge_optimizations() 

    print("Edge Optimization Results:") 

    for optimization, metrics in results.items(): 

        print(f"{optimization}: {metrics}") 