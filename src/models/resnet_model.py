"""

ResNet50 with pretrained ImageNet weights, adapted for GTSRB
Input - 32x32 normalized RGB images
The model internally resizes to 128x128 and applies ResNet50's expected preporcessing
before passing to conv base
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

NUM_CLASSES = 43
INPUT_SHAPE = (32, 32, 3)
RESIZED_SHAPE = (128, 128)

def build_resnet50_model(freeze_base=True):
    """
    Builds the ResNet50-based model for GTSRB

    Args:
        freeze_base: if true, the pretrained base is frozen for phase 1 (feature extraction)
                     if false, all layers are trainable for phase 2 (fine tuning)
    """

    inputs = layers.Input(shape=INPUT_SHAPE)

    # resize 32x32 to 128x128
    x = layers.Resizing(*RESIZED_SHAPE)(inputs)

    # resnet50 preprocessing: scale [0, 1] to [0, 255], then mean subtract
    # keeps RGB order also
    x = layers.Lambda(lambda t: preprocess_input(t*255.0), name = 'resnet50_preprocessing')(x)

    # load resnet50 with imagenet weights, no classifier head
    base = ResNet50(
        weights='imagenet',
        include_top = False,
        input_shape = (*RESIZED_SHAPE, 3),
    )
    base.trainable = not freeze_base

    # noe we pass training=False duing phase 1 so BatchNorm uses running stats and doesn't update them
    x = base(x, training = False if freeze_base else None)

    # classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='resnet50_gtsrb')
    return model


def compile_phase1(model, learning_rate = 1e-3):
    """ phase 1 training, head only with normal learning rate"""
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
    )
    return model

def compile_phase2(model, learning_rate = 1e-5):
    """ phase 2 training, fine tuning model. 100x smaller learning rate """
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
    )
    return model