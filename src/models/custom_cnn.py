"""
VGG-style CNN. Baseline model for comparison against transfer learning approaches
"""

import tensorflow as tf
from tensorflow.keras import layers, models

NUM_CLASSES = 43
INPUT_SHAPE = (32, 32, 3)

def build_custom_cnn():
    """
    Architcture: 3 conv blocks (each with 2 conv layers + BN + pool)
    followed by global average pooling and a dense classifier head
    """

    model = models.Sequential([
        layers.Input(shape = INPUT_SHAPE),
        
        # Block 1: 32 filters, output (16, 16, 32) after pool
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2: 64 filters, output (8, 8, 64) after pool
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3: 128 filters, output (4, 4, 128) after pool
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classifier head
        layers.GlobalAveragePooling2D(),    # → vector of length 128
        layers.Dropout(0.5),                # regularization
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ], name = 'custom_cnn_baseline')

    return model

def compile_model(model, learning_rate = 0.001):
    """
    Compile with Adam optimizer and sparse categorical crossentropy
    
    using sparse_* because labels are integer-encoded
    """

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
    )

    return model