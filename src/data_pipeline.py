"""
Builds tf.data pipelines for training, validation, and test sets.
Augmentation is applied to the training pipeline only.

Expects integer labels (shape (N,)). Use sparse_categorical_crossentropy
as the loss when training.
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split

BATCH_SIZE = 64
SHUFFLE_BUFFER = 10000
VAL_SPLIT = 0.2
RANDOM_SEED = 42

def build_augmentation_layer():
    """
    no flips and small rotations due to GTSRB dataset
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.03, seed=RANDOM_SEED),
        tf.keras.layers.RandomTranslation(0.1, 0.1, seed=RANDOM_SEED),
        tf.keras.layers.RandomZoom(0.1, seed=RANDOM_SEED),
        tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0), seed=RANDOM_SEED),
    ], name = "gtsrb_augmentation")


def build_pipelines(X_train_full, y_train_full, X_test, y_test):
    """
    splits training data into train/val and builds tf.data pipelines for all 3 splits
    returns (train_ds, val_ds, test_ds)
    """

    # stratified train/val split BEFORE building any pipeline
    # stratify preserves class distribution (imbalance exists in data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SPLIT, stratify=y_train_full, random_state=RANDOM_SEED
    )

    #augmentation layer
    augmentation = build_augmentation_layer()

    def augment_fn(x, y):
        return augmentation(x, training=True), y

    # training pipeline: shuffle -> batch -> augment -> prefetch
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(SHUFFLE_BUFFER, seed=RANDOM_SEED)
        .batch(BATCH_SIZE)
        .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # validation pipeline: no shuffle, no augment, just batch and prefetch
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # test pipeline
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds