"""
    This module contains the p53 MLP model code.
    6. This module should be the sixth module in the pipeline.
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, Dropout
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Input
from src import MODELS_DIR, P53_MODEL_NAME
from src.models_usage import save_model as general_save_model

def p53_train_model(X_train, y_train, X_test, y_test, pfam = False) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
        Create and train the p53 MLP model.
        Model architecture:
            - Input layer
            - 64 units, ReLU activation
            - Dropout 0.3
            - 128 units, ReLU activation
            - Dropout 0.3
            - 64 units, ReLU activation
            - Dropout 0.3
            - Output layer, 3 units, softmax activation

        Parameters:
            X_train: The training features.
            y_train: The training labels.
            X_test: The test features.
            y_test: The test labels.
            pfam: If True, the model will be trained with Pfam data. Default is False.
        Returns:
            The trained model and the training history.
    """

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        #Dense(64, activation='relu'),
        #Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),  # Prevent overfitting
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')  # Adjust the output units for the number of classes
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # For multiclass classification
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        X_train, tf.keras.utils.to_categorical(y_train),  # Convert labels to one-hot
        validation_data=(X_test, tf.keras.utils.to_categorical(y_test)),
        epochs=50,  # Adjusted based on performance
        batch_size=32,
        verbose=1
    )

    # Evaluate the model
    #print("\nTest Evaluation:")
    #print(model.evaluate(X_test, tf.keras.utils.to_categorical(y_test)))

    return model, history




def save_model(model: tf.keras.Model, name: str = P53_MODEL_NAME):
    """
        Use the general save_model function to save the model in models_usage.py module.
        Save the model to a file. 
        If a model with the same name exists, it will be renamed with a .bak extension.
        Files are saved in the models directory.
        Models are saved in both .h5 and .keras formats.

        Parameters:
            model: The model to save.
            name: The name of the model
        Returns:
            str: The path where the model was saved. (keras format) or None if an error occurred.
    """
    return general_save_model(model, name)


def retrain_model_to_save(model, X_train, y_train, pfam = False) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
        Train a model with all the given data in order to save it.
        It does not split the data into training and test sets.
        It doesn't save the model.

        Parameters:
            model: The model to train.
            X_train: The training features.
            y_train: The training labels.
            pfam: If True, the model will be trained with Pfam data. Default is False.
        Returns:
            The trained model and the training history.
    """

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train, 
        tf.keras.utils.to_categorical(y_train), 
        epochs=50, 
        batch_size=32, 
        verbose=1
    )

    return model, history


def model_predict(model: tf.keras.Model, X: pd.DataFrame) -> tuple[tf.Tensor, np.ndarray]:
    """
        Predict the labels for the given features.
        Parameters:
            model: The trained model.
            X: The features to predict.
        Returns:
            Predicted probabilities: as a Tensor
            Predicted labels: as a numpy array of indexes
    """
    # Convert to Tensor
    X_tensor = tf.convert_to_tensor(X.values, dtype=tf.float32)

    # Predict Probabilities
    probabilities = model.predict(X_tensor)

    return probabilities, tf.argmax(probabilities, axis=1).numpy()