"""
    This module contains the model for the HRAS-6 dataset using Transfer Learning.
    6. This should be the sixth module to be executed for Hras.
"""

import tensorflow as tf
from keras import Sequential
from keras.api.layers import Dense, Dropout, Input
from keras.api.models import Sequential


def hras_transfer_model(X_train, X_test, y_train, y_test, original_model: tf.keras.Model) \
                    -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
        Train the HRAS model using Transfer Learning.
        1. Train the model with the HRAS data.
        2. Freeze all layers but the last
        3. Remove the last layer from the model
        4. Add a new layer
        5. Compile the model
        6. Train the model
        7. Return the new model and the training history.

        Parameters:
            X_train: The training features.
            X_test: The test features.
            y_train: The training labels.
            y_test: The test labels.
            original_model: The original model to use for transfer learning.

        Returns:
            tuple: The trained model and the training history
    """

    
    for layer in original_model.layers[:-1]:  # Freeze all layers but the last
        layer.trainable = False

    # Remove the last layer from the model
    original_model.pop()

    # Add a new layer
    original_model.add(Dense(3, activation='softmax'))

    original_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = original_model.fit(
        X_train, tf.keras.utils.to_categorical(y_train),
        validation_data=(X_test, tf.keras.utils.to_categorical(y_test)),
        epochs=30,
        batch_size=16,
        verbose=1
    )

    return original_model, history


def hras_transfer_model_v2(X_train, X_test, y_train, y_test, original_model: tf.keras.Model) \
                    -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
        Train the HRAS model using Transfer Learning.
        1. Train the model with the HRAS data.
        2. Freeze all layers but the last
        3. Remove the last layer from the model and
        4. Add 2 new layers: Output layer, 3 units, softmax activation
        5. Set learning rate to 0.001 for last but one layer
        5. Compile the model
        6. Train the model
        7. Return the new model and the training history.
    """

    for layer in original_model.layers[:-2]:  # Freeze all layers but the last 2
        layer.trainable = False
    
    # Remove the last layer from the model
    original_model.pop()

    # Set the learning rate for the last but one layer
    original_model.layers[-1].learning_rate = 0.005

    # Add last layer
    original_model.add(Dense(3, activation='softmax'))

    # Compile the model
    original_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = original_model.fit(
        X_train, tf.keras.utils.to_categorical(y_train),
        validation_data=(X_test, tf.keras.utils.to_categorical(y_test)),
        epochs=30,
        batch_size=16,
        verbose=1
    )

    return original_model, history



def retrain_hras_model_to_save(X_train, y_train, model):
    """
        Retrain the HRAS model with the full dataset in order to save it.
        Parameters:
            X_train: The full training features.
            y_train: The full training labels.
            model: The model to retrain.
        Returns:
            The retrained model.
    """

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, tf.keras.utils.to_categorical(y_train),
        epochs=30,
        batch_size=16,
        verbose=1
    )

    return model