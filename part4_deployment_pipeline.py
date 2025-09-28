import tensorflow as tf 

import tensorflow_model_optimization as tfmot

import json 

import os

import numpy as np 

from abc import ABC, abstractmethod 

from dataclasses import dataclass 

from typing import Dict, List, Tuple, Any 

import time

from logger_utils import get_simple_logger

from part1_baseline import load_and_preprocess_data
from streamlined_analysis import streamlined_model_analysis
 

@dataclass 

class DeploymentTarget: 

    """Configuration for different deployment targets.""" 

    name: str 

    max_model_size_mb: float 

    max_latency_ms: float 

    max_memory_mb: float 

    power_budget_mw: float 

    compute_capability: str  # 'cloud', 'edge', 'tiny' 

 

@dataclass 

class OptimizationResult: 

    """Results from model optimization.""" 

    model_path: str 

    accuracy: float 

    model_size_mb: float 

    estimated_latency_ms: float 

    memory_usage_mb: float 

    optimization_strategy: str 

def measure_model_accuracy_latency(model, x_test, y_test):
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    part4_logger.info(f"Input type: {input_details['dtype']}, Quantization: {input_details['quantization']}")
    part4_logger.info(f"Output type: {output_details['dtype']}, Quantization: {output_details['quantization']}")
    correct_predictions = 0
    is_int8 = input_details['dtype'] == np.int8
    if is_int8:
        input_data_scale = input_details['quantization'][0]
        input_data_zero_point = input_details['quantization'][1]
        
    inference_time = 0
    for i in range(x_test.shape[0]):
        input_data = x_test[i:i+1].astype(np.float32)
        if is_int8:
            input_data = (input_data / input_data_scale + input_data_zero_point).astype(np.int8)
        time_start = time.time()
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])
        predicted_class = tf.argmax(output_data[0], axis=0).numpy()
        time_end = time.time()
        if predicted_class == y_test[i]:
            correct_predictions += 1
        inference_time += time_end - time_start
    latency = inference_time / x_test.shape[0] * 1000 # ms
    accuracy = correct_predictions / x_test.shape[0]
    return accuracy, latency
class ModelOptimizer(ABC): 

    @abstractmethod 

    def optimize(self, model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult: 

        pass 


class CloudOptimizer(ModelOptimizer): 

    def optimize(self, model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult: 

        # TODO: Implement cloud-specific optimizations 
        # Focus on throughput and multi-GPU scaling 
        part4_logger.info("Optimizing for cloud")
        x_train, y_train, x_test, y_test = load_and_preprocess_data()
        
        tf_strategy = tf.distribute.MirroredStrategy()
        with tf_strategy.scope():
            new_model = tf.keras.models.Sequential()
            for layer in model.layers:
                config = layer.get_config()
                new_layer = layer.__class__.from_config(config)
                new_layer._dtype_policy = tf.keras.mixed_precision.Policy('mixed_float16')
                new_model.add(new_layer)
            new_model.set_weights(model.get_weights())
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        # Use from_logits=True to match the baseline model architecture
        new_model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        part4_logger.info("Distributed mixed precision model compiled")
        part4_logger.info(f"Distributed Strategy: {tf_strategy.__class__.__name__}")
        for layer in new_model.layers:
            part4_logger.info(f"Layer: {layer.name}, dtype: {layer.dtype_policy}")
        
        part4_logger.info("Training distributed mixed precision model")
        time_start = time.time()
        
        # Add callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        new_model.fit(
            x_train, y_train, 
            epochs=20, 
            batch_size=64, 
            validation_data=(x_test, y_test), 
            callbacks=callbacks,
            verbose=2
        )
        time_end = time.time()
        part4_logger.info(f"Time taken to train distributed mixed precision model: {time_end - time_start} seconds")
        
        model_path = 'part4_deployment_pipeline/distributed_mixed_precision_model.keras'
        new_model.save(model_path)
        
        stream_line_analysis = streamlined_model_analysis(new_model, x_test, y_test, 64, model_path, part4_logger)
        return OptimizationResult(
            model_path=model_path,
            accuracy=stream_line_analysis['performance_speed']['test_accuracy'],
            model_size_mb=stream_line_analysis['architecture_memory']['file_size_mb'],
            estimated_latency_ms=stream_line_analysis['performance_speed']['single_sample_time_ms'],
            memory_usage_mb=stream_line_analysis['architecture_memory']['training_memory_mb'],
            optimization_strategy=["mixed_precision", "distributed"]
        )

class EdgeOptimizer(ModelOptimizer): 

    def optimize(self, model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult: 

        # TODO: Implement edge-specific optimizations 

        # Focus on latency and memory efficiency 
        part4_logger.info("Optimizing for edge")
        x_train, y_train, x_test, y_test = load_and_preprocess_data()
        
        part4_logger.info("Implementing pruning")
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=0.5,
                begin_step=0,
                end_step=100000
            )
        }
        new_model = tf.keras.models.clone_model(model)
        new_model = tfmot.sparsity.keras.prune_low_magnitude(new_model, **pruning_params)
        new_model.compile(
            optimizer='adam', 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            metrics=['accuracy']
        )
        part4_logger.info("Finetuning pruned model")
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
        ]
        new_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=2, callbacks=callbacks)
        part4_logger.info("Applying dynamic range quantization")
        new_model = tf.lite.TFLiteConverter.from_keras_model(new_model)
        new_model.optimizations = [tf.lite.Optimize.DEFAULT]
        new_model = new_model.convert()
        
        model_path = 'part4_deployment_pipeline/edge_optimized_model.tflite'
        part4_logger.info(f"Saving edge optimized model to: {model_path}")
        with open(model_path, 'wb') as f:
            f.write(new_model)
        
        part4_logger.info("Measuring model accuracy and latency")
        
        accuracy, latency = measure_model_accuracy_latency(new_model, x_test, y_test)
        
        model_size = os.path.getsize(model_path) / 1024 / 1024
        per_sample_data_size = np.prod(x_test.shape[1:]) * 4 / 1024 / 1024 # MB
        
        part4_logger.info(f"Model accuracy: {accuracy}")
        part4_logger.info(f"Model latency: {latency} ms")
        part4_logger.info(f"Model size: {model_size} MB")
        
        return OptimizationResult(
            model_path=model_path,
            accuracy=accuracy,
            model_size_mb=model_size,
            estimated_latency_ms=latency,
            memory_usage_mb=model_size+per_sample_data_size,
            optimization_strategy=['0.5_pruning', 'dynamic_range_quantization']
        )

 

class TinyMLOptimizer(ModelOptimizer): 

    def optimize(self, model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult: 

        # TODO: Implement tinyML-specific optimizations 

        # Focus on extreme resource constraints 
        part4_logger.info("Optimizing for tinyML")
        x_train, y_train, x_test, y_test = load_and_preprocess_data()
        
        part4_logger.info("Implementing architecture optimization")
        architecture={'depth': 2, 'width': 32, 'kernel_size': 3}
        input_shape = x_train.shape[1:]
        num_classes = 10
        current_filters = architecture['width']
        new_model = tf.keras.models.Sequential()
        new_model.add(tf.keras.layers.Input(shape=input_shape))
        for block_idx in range(architecture['depth']):
            new_model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=architecture['kernel_size'], padding='same', depth_multiplier=1))
            new_model.add(tf.keras.layers.BatchNormalization())
            new_model.add(tf.keras.layers.ReLU(max_value=6))
            new_model.add(tf.keras.layers.Conv2D(current_filters, kernel_size=1, padding='same'))
            new_model.add(tf.keras.layers.BatchNormalization())
            new_model.add(tf.keras.layers.ReLU(max_value=6))
            new_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            current_filters = min(current_filters * 2, 512)
        new_model.add(tf.keras.layers.Flatten())
        new_model.add(tf.keras.layers.Dropout(0.5))
        new_model.add(tf.keras.layers.Dense(256))
        new_model.add(tf.keras.layers.ReLU(max_value=6))
        new_model.add(tf.keras.layers.Dropout(0.3))
        new_model.add(tf.keras.layers.Dense(num_classes))
        new_model.compile(
            optimizer='adam', 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            metrics=['accuracy']
        )
        new_model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test), verbose=2)

        part4_logger.info("Implementing pruning")
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=0.75,
                begin_step=0,
                end_step=100000
            )
        }
        new_model = tfmot.sparsity.keras.prune_low_magnitude(new_model, **pruning_params)
        new_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        part4_logger.info("Finetuning pruned model")
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
        ]
        new_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=2, callbacks=callbacks)
        part4_logger.info("Applying float16 quantization")
        new_model = tf.lite.TFLiteConverter.from_keras_model(new_model)
        new_model.optimizations = [tf.lite.Optimize.DEFAULT]
        new_model.target_spec.supported_types = [tf.float16]
        new_model = new_model.convert()
        
        model_path = 'part4_deployment_pipeline/tinyML_optimized_model.tflite'
        part4_logger.info(f"Saving tinyML optimized model to: {model_path}")
        with open(model_path, 'wb') as f:
            f.write(new_model)
        
        part4_logger.info("Measuring model accuracy and latency")
        
        accuracy, latency = measure_model_accuracy_latency(new_model, x_test, y_test)
        
        model_size = os.path.getsize(model_path) / 1024 / 1024
        per_sample_data_size = np.prod(x_test.shape[1:]) * 1 / 1024 / 1024 # MB
        
        part4_logger.info(f"Model accuracy: {accuracy}")
        part4_logger.info(f"Model latency: {latency} ms")
        part4_logger.info(f"Model size: {model_size} MB")
        
        return OptimizationResult(
            model_path=model_path,
            accuracy=accuracy,
            model_size_mb=model_size,
            estimated_latency_ms=latency,
            memory_usage_mb=model_size+per_sample_data_size,
            optimization_strategy=['architecture_optimization', '0.75_pruning', 'float16_quantization']
        )

 

class MultiScaleDeploymentPipeline: 

    """ 

    Automated pipeline for optimizing models across different deployment scales. 

    """ 

     

    def __init__(self): 

        self.optimizers = { 

            'cloud': CloudOptimizer(), 

            'edge': EdgeOptimizer(), 

            'tiny': TinyMLOptimizer() 

        } 

         

        # Define deployment targets 

        self.targets = { 

            'cloud_server': DeploymentTarget( 

                name='cloud_server', 

                max_model_size_mb=1000.0, 

                max_latency_ms=100.0, 

                max_memory_mb=8000.0, 

                power_budget_mw=50000.0, 

                compute_capability='cloud' 

            ), 

            'edge_device': DeploymentTarget( 

                name='edge_device', 

                max_model_size_mb=50.0, 

                max_latency_ms=200.0, 

                max_memory_mb=512.0, 

                power_budget_mw=2000.0, 

                compute_capability='edge' 

            ), 

            'microcontroller': DeploymentTarget( 

                name='microcontroller', 

                max_model_size_mb=1.0, 

                max_latency_ms=1000.0, 

                max_memory_mb=64.0, 

                power_budget_mw=10.0, 

                compute_capability='tiny' 

            ) 

        } 

     

    def optimize_for_all_targets(self, baseline_model_path: str) -> Dict[str, OptimizationResult]: 

        """ 

        Optimize baseline model for all deployment targets. 

         

        Args: 

            baseline_model_path: Path to baseline Keras model 

             

        Returns: 

            Dictionary mapping target names to optimization results 

        """ 

        baseline_model = tf.keras.models.load_model(baseline_model_path) 

        results = {} 

         

        for target_name, target_config in self.targets.items(): 

            optimizer = self.optimizers[target_config.compute_capability] 

            results[target_name] = optimizer.optimize(baseline_model, target_config) 

             

        return results 

     

    def analyze_scaling_trade_offs(self, results: Dict[str, OptimizationResult]) -> Dict[str, Any]: 

        """ 

        Analyze trade-offs across different deployment scales. 

         

        Args: 

            results: Optimization results from optimize_for_all_targets 

             

        Returns: 

            Comprehensive analysis of scaling trade-offs 

        """ 

        analysis = {} 

         

        # TODO: Calculate Pareto frontier of accuracy vs efficiency 
        points = []
        for target_name, result in results.items():
            points.append({
                'target': target_name,
                'accuracy': result.accuracy,
                'latency': result.estimated_latency_ms,
            })
        pareto_frontier = []
        for point in points:
            is_pareto = True
            for other_point in points:
                if point == other_point:
                    continue
                if point['accuracy'] < other_point['accuracy'] and point['latency'] > other_point['latency']:
                    is_pareto = False
                    break
            if is_pareto:
                pareto_frontier.append(point)
        analysis["pareto_frontier"] = pareto_frontier

        # TODO: Analyze scaling efficiency across deployment targets 
        # TODO: Identify bottlenecks for each deployment scenario 
        for target_name, target_config in self.targets.items():
            result = results[target_name]
            # Initialize the analysis dictionary for this target
            analysis[target_name] = {}
            
            normalized_values = {}
            normalized_values['latency'] = result.estimated_latency_ms / target_config.max_latency_ms
            normalized_values['memory'] = result.memory_usage_mb / target_config.max_memory_mb
            normalized_values['model_size'] = result.model_size_mb / target_config.max_model_size_mb
            accuracy = result.accuracy
            scaling_efficiency = accuracy / (normalized_values['latency'] + normalized_values['memory'] + normalized_values['model_size'])
            bottleneck = None
            for key, value in normalized_values.items():
                if value == max(normalized_values.values()):
                    bottleneck = key
                    break
            analysis[target_name]["scaling_efficiency"] = scaling_efficiency
            analysis[target_name]["bottleneck"] = bottleneck
            # Add the raw metrics for deployment recommendations
            analysis[target_name]["model_size_mb"] = result.model_size_mb
            analysis[target_name]["latency"] = result.estimated_latency_ms
            analysis[target_name]["memory_usage_mb"] = result.memory_usage_mb

        # TODO: Recommend optimal deployment strategy for different use cases 
        for target_name, target_config in self.targets.items():
            if target_name not in results:
                continue
            result = results[target_name]
            part4_logger.info(f"For use case {target_name}, deploy with {result.optimization_strategy} to achieve the best performance")

        return analysis 

     

    def generate_deployment_recommendations(self, analysis: Dict[str, Any]) -> List[str]: 

        """ 

        Generate actionable deployment recommendations. 

         

        Args: 

            analysis: Results from analyze_scaling_trade_offs 

             

        Returns: 

            List of deployment recommendations 

        """ 

        recommendations = [] 
         
        # TODO: Analyze which targets meet performance requirements 
        target_recommendations = {}
        for target_name, target_config in self.targets.items():
            if target_name not in analysis:
                part4_logger.warning(f"No analysis found for target: {target_name}")
                continue
            result = analysis[target_name]
            # Check if the result has the required keys
            if 'model_size_mb' in result and 'latency' in result and 'memory_usage_mb' in result:
                if result['model_size_mb'] <= target_config.max_model_size_mb and result['latency'] <= target_config.max_latency_ms \
                and result['memory_usage_mb'] <= target_config.max_memory_mb:
                    target_recommendations[target_name] = target_name
        part4_logger.info(f"The targets that meet the performance requirements are: {target_recommendations}")
        
        # Add target recommendations to the list
        if target_recommendations:
            recommendations.append(f"Recommended targets: {target_recommendations}")

        # TODO: Identify opportunities for cascaded deployment 
        cascaded_opportunities = """
        Cascaded deployment is used when we require high accuracy and low energy consumption,
        which most used in the edge devices or microcontrollers. We can use a small model optimized with tinyML optimization
        combine a larger model optimized with edge optimization methods to achieve the best performance. Small model usually have
        low latency and low energy consumption, we use it to preprocess data and trigger the larger model to make more comfidence prediction.
        """
        recommendations.append(cascaded_opportunities)
        # TODO: Suggest hybrid deployment strategies 
        hybrid_deployment = f"""
        We can use the pareto frontiers to decide which models is the best for each target. Model with higher accuracy and model size
        is better for the cloud server, model with lower latency and model size is better for the edge devices, model with lowest latency and memory usage
        is better for the microcontrollers.
        Pareto frontier: {analysis['pareto_frontier']}
        """
        recommendations.append(hybrid_deployment)
        # TODO: Recommend development priorities
        # Collect bottlenecks from all targets
        bottlenecks = {}
        for target_name, target_config in self.targets.items():
            if target_name in analysis and 'bottleneck' in analysis[target_name]:
                bottlenecks[target_name] = analysis[target_name]['bottleneck']
        
        development_priorities = f"""
        We should prioritize the development of each bottlenecks for each target. 
        For example, if the bottleneck is latency, we should prioritize reducing the latency.
        Current bottlenecks: {bottlenecks}
        """
        recommendations.append(development_priorities)  

        return recommendations 

 

def run_multi_scale_optimization(): 

    """ 

    Execute complete multi-scale optimization pipeline. 

    """ 

    pipeline = MultiScaleDeploymentPipeline() 

     

    # Optimize for all targets 

    results = pipeline.optimize_for_all_targets('part1_baseline/best_model.keras') 

     

    # Analyze trade-offs 

    analysis = pipeline.analyze_scaling_trade_offs(results) 

     

    # Generate recommendations 

    recommendations = pipeline.generate_deployment_recommendations(analysis) 

     

    # Generate comprehensive report 

    report = { 

        'optimization_results': results, 

        'scaling_analysis': analysis, 

        'deployment_recommendations': recommendations 

    } 

     

    # Save report 

    with open('multi_scale_optimization_report.json', 'w') as f: 

        json.dump(report, f, indent=2, default=str) 

     

    return report 

 

if __name__ == "__main__": 
    os.makedirs('part4_deployment_pipeline', exist_ok=True)
    part4_logger = get_simple_logger('part4_deployment_pipeline', 'part4_deployment_pipeline/part4_terminal.log')

    report = run_multi_scale_optimization() 

    part4_logger.info("Multi-Scale Optimization Complete!") 

    part4_logger.info(f"Report saved to: multi_scale_optimization_report.json") 