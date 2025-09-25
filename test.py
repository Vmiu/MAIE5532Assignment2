import tensorflow as tf
import numpy as np
from itertools import product
import time

class EdgeOptimizer:
    def __init__(self, baseline_model_path, x_train, y_train, x_test, y_test):
        self.baseline_model = tf.keras.models.load_model(baseline_model_path)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def implement_neural_architecture_search(self):
        """
        Implement simplified NAS for finding optimal edge architecture.
        
        Returns:
            tuple: (best_architecture, search_results)
        """
        # Define search space based on baseline architecture variations
        search_space = {
            'num_blocks': [1,2,3],  # Number of conv blocks (baseline has 3)
            'base_filters': [16, 32, 48],  # Starting filter count (baseline starts with 32)
            'use_double_conv': [True, False],  # Whether to use 2 conv layers per block (baseline uses 2)
            'dense_units': [128, 256, 512],  # Dense layer size (baseline uses 256)
            'use_global_pooling': [True, False]  # GlobalAvgPool vs Flatten (baseline uses GlobalAvgPool)
        }
        
        # Generate architecture candidates (random sampling for efficiency)
        architectures = []
        np.random.seed(42)
        
        # Include baseline architecture as reference
        baseline_arch = {
            'num_blocks': 3,
            'base_filters': 32,
            'kernel_size': 3,
            'use_double_conv': True,
            'dense_units': 256,
            'dropout_dense': 0.5,
            'dropout_final': 0.3,
            'use_global_pooling': True
        }
        architectures.append(baseline_arch)
        
        # Generate 99 more random architectures
        for _ in range(99):
            arch = {}
            for key, value_list in search_space.items():
                arch[key] = np.random.choice(value_list)
            architectures.append(arch)
        
        # Evaluate architectures
        search_results = []
        best_architecture = None
        best_score = -np.inf
        
        print(f"Evaluating {len(architectures)} architectures...")
        print("Architecture 1 is the baseline model for reference.")
        
        for i, arch in enumerate(architectures):
            print(f"Architecture {i+1}/{len(architectures)}: {arch}")
            
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
            metrics['is_baseline'] = (i == 0)
            search_results.append(metrics)
            
            # Update best architecture
            if composite_score > best_score:
                best_score = composite_score
                best_architecture = arch
                
        # Sort results by composite score
        search_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Find Pareto-optimal architectures
        pareto_optimal = self._find_pareto_optimal(search_results)
        
        return best_architecture, {
            'all_results': search_results,
            'pareto_optimal': pareto_optimal,
            'best_composite_score': best_score,
            'baseline_performance': next(r for r in search_results if r.get('is_baseline', False))
        }
    
    def _evaluate_architecture(self, architecture):
        """
        Architecture evaluation function based on baseline model structure
        
        Args:
            architecture: Dictionary defining the architecture parameters
            
        Returns:
            dict: Metrics including accuracy, model size, and inference time
        """
        # Build model with given architecture
        model = self._build_model(architecture)
        
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
    
    def _build_model(self, architecture):
        """Build model based on architecture parameters, following baseline structure"""
        input_shape = self.x_train.shape[1:]
        num_classes = 10  # Based on baseline model
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))
        
        # Build convolutional blocks similar to baseline
        current_filters = architecture['base_filters']
        
        for block_idx in range(architecture['num_blocks']):
            # First conv layer in block
            model.add(tf.keras.layers.Conv2D(
                filters=current_filters,
                kernel_size=architecture['kernel_size'],
                padding='same'
            ))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())
            
            # Second conv layer in block (if using double conv)
            if architecture['use_double_conv']:
                model.add(tf.keras.layers.Conv2D(
                    filters=current_filters,
                    kernel_size=architecture['kernel_size'],
                    padding='same'
                ))
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.ReLU())
            
            # Max pooling after each block
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
            
            # Double filters for next block (similar to baseline: 32->64->128)
            current_filters = min(current_filters * 2, 512)  # Cap at 512 to prevent explosion
        
        # Classifier head similar to baseline
        if architecture['use_global_pooling']:
            model.add(tf.keras.layers.GlobalAveragePooling2D())
        else:
            model.add(tf.keras.layers.Flatten())
        
        model.add(tf.keras.layers.Dropout(architecture['dropout_dense']))
        model.add(tf.keras.layers.Dense(architecture['dense_units'], activation='relu'))
        model.add(tf.keras.layers.Dropout(architecture['dropout_final']))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
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