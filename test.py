import tensorflow as tf

def convert_to_mixed_precision(model, target_policy='mixed_float16'):
    """Convert a model to mixed precision by recreating layers from original model config"""
    original_policy = tf.keras.mixed_precision.global_policy()
    
    try:
        # Set global policy to mixed precision
        tf.keras.mixed_precision.set_global_policy(target_policy)
        
        # Create new Sequential model - this will inherit the global policy
        mixed_precision_model = tf.keras.Sequential()
        
        # Recreate each layer from the original model's configuration
        for i, layer in enumerate(model.layers):
            # Get the layer configuration
            config = layer.get_config()
            
            # Create the layer with the same configuration (including activation)
            new_layer = layer.__class__.from_config(config)
            
            # Explicitly set the dtype policy to mixed precision
            new_layer._dtype_policy = tf.keras.mixed_precision.Policy(target_policy)
            
            # Copy the trained weights from the original layer
            new_layer.set_weights(layer.get_weights())
            
            # Add the layer to the new model
            mixed_precision_model.add(new_layer)
        
        return mixed_precision_model
        
    finally:
        # Restore original policy
        tf.keras.mixed_precision.set_global_policy(original_policy)

# Load original model
original_policy = tf.keras.mixed_precision.global_policy()
print(f"Original policy: {original_policy}")
model = tf.keras.models.load_model('part1_baseline/baseline_model.keras')
print(model.summary())

# Convert to mixed precision
mixed_precision_model = convert_to_mixed_precision(model, 'mixed_float16')

# Set up optimizer with loss scaling
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

# Compile the mixed precision model
mixed_precision_model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

for layer in mixed_precision_model.layers:
    print(layer.name)
    print(layer.dtype_policy)
