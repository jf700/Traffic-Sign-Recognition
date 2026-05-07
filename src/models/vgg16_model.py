"""

VGG16 with pretrained ImageNet weights, adapted for GTSRB
Input is 32x32 normalized RGB images from the pipeline already created

model resizes to 64x64 and applies the VGG16's expected preprocessing before training
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

NUM_CLASSES = 43
INPUT_SHAPE = (32, 32, 3)
RESIZED_SHAPE = (64, 64)

def build_vgg16_model(freeze_base=True):
    """
    Builds VGG16-based model

    Args:
        freeze_base: If true, VGG16's pretrained layers are frozen (feature extraction)
                     if false, all layers are trainable (fine tuning)
    """

    # input layer matches existing pipeline: 32x32 RGB in [0, 1]
    inputs = layers.input(shape=INPUT_SHAPE)

    # resize to 64x64
    x = layers.Resizing(*RESIZED_SHAPE)(inputs)

    # convert [0, 1] floats to VGG16 expected format
    # preprocess_input expects [0, 255] BGR with mean-subtraction
    # so multiply by 255 first, then let preprocess_input do the rest
    x = layers.Lambda(lambda t: preprocess_input(t * 255.0), name = 'vgg16_preprocessing')(x)

    # load VGG16 with ImageNet weights, no classifier head
    base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape = (*RESIZED_SHAPE, 3),
    )
    base.trainable = not freeze_base

    x = base(x, training = False if freeze_base else None)

    # classifer head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs, outputs, name = 'vgg16_gtsrb')
    return model


def compile_phase1(model, learning_rate=1e-3):
    """ ONLY training head, normal learning rate"""
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
    )
    return model

def compile_phase2(model, learning_rate = 1e-4):
    """ FINE TUNE model, 10x less learning rate """
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
    )
    return model