"""
    This module contains the model for the HRAS-6 dataset using Transfer Learning.
    6. This should be the sixth module to be executed for Hras.
"""

import tensorflow as tf
import pandas as pd 
import numpy as np
from keras import Sequential, Model
from keras.api.layers import Dense, Dropout, Input
from keras.api.callbacks import EarlyStopping

from src import MODELS_DIR, HRAS_MODEL_NAME
from src.models_usage import save_model as general_save_model
from src.p53.p53_6_model_v2 import N5_columns, N4_columns, AA21_columns

# Use V1
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
        verbose=1,
        callbacks=[__early_stopping()]
    )

    return original_model, history

# Uses V2
def hras_transfer_model_v2(X_train, X_test, y_train, y_test, 
                           original_model:tf.keras.Model, pfam = True) \
                    -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
        Train the HRAS model using Transfer Learning from P53 Model V2.
        1. Train the model with the HRAS data.
        2. Freeze all layers but the last
        3. Remove the last layer from the model and
        4. Add 2 new layers: Output layer, 3 units, softmax activation
        5. Set learning rate to 0.001 for last but one layer
        5. Compile the model
        6. Train the model
        7. Return the new model and the training history.

        Parameters:
            X_train: The training features.
            X_test: The test features.
            y_train: The training labels.
            y_test: The test labels.
            original_model: The original model to use for transfer learning.
            pfam: If True, the model will be trained with Pfam data. Default is True.
        
        Returns:
            tuple: The trained model and the training history
    """

     # Freeze all layers except the last one
    for layer in original_model.layers[:-1]:
        layer.trainable = False
    for layer in original_model.layers[-2:]:
        layer.trainable = True # Unfreeze the last two layers

    # Replace the output layer
    x = original_model.layers[-2].output  # Take the output of the second-to-last layer
    new_output = Dense(3, activation='softmax', name='hras_output')(x)

    # Create a new model with the modified output
    new_model = Model(inputs=original_model.input, outputs=new_output)

    # Compile the new model
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.010), # TODO: Check learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Prepare the data dictionary
    X_train_dict = input_dict_prepare(X_train, pfam)
    X_test_dict = input_dict_prepare(X_test, pfam)

    # Train the model
    history = new_model.fit(
        x=X_train_dict,
        y=tf.keras.utils.to_categorical(y_train),
        validation_data=(X_test_dict, tf.keras.utils.to_categorical(y_test)),
        epochs=50,
        batch_size=16,
        verbose=1,
        callbacks=[__early_stopping()]
    )

    return new_model, history





def retrain_hras_model_to_save(X_train, y_train, model, 
                               use_P53_v2 = True, pfam = True) -> \
                    tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
        Retrain the HRAS model with the full dataset in order to save it.
        Parameters:
            X_train: The full training features.
            y_train: The full training labels.
            model: The model to retrain.
            use_P53_v2: If True, the model is trained from P53 V2 Model. Default is True.
            pfam: If True, the model will be trained with Pfam data. Default is True.
        Returns:
            The retrained model and the training history.
    """

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    if use_P53_v2:
        X_train = input_dict_prepare(X_train, pfam)


    history = model.fit(
        x=X_train, 
        y=tf.keras.utils.to_categorical(y_train),
        epochs=50,
        batch_size=16,
        verbose=1,
        callbacks=[__early_stopping()]
    )

    return model, history



# -------------------------------------------- AIDE FUNCTIONS -------------------------------------------- #

def input_dict_prepare(X: pd.DataFrame, pfam=False) -> dict:
    """
    Prepare the dictionary of data for the model.
    
    Parameters:
        X: pd.DataFrame
            The features to predict.
        pfam: bool
            If True, the model will be trained with Pfam data. Default is False.
    
    Returns:
        dict: The dictionary of data for the model.
    """
    # Initialize the dictionary with encoded columns
    X_dict = {column + '_Encoded': X[column + '_Encoded'] \
              for column in N5_columns + N4_columns + AA21_columns}
    
    # Add numerical inputs based on pfam flag
    if pfam:
        X_dict['Position and Conservation'] = X[['cDNA_Position', 'Conservation']].to_numpy()
    else:
        X_dict['cDNA_Position'] = X['cDNA_Position'].to_numpy()
    
    return X_dict


def __early_stopping():
    """
        Create an EarlyStopping callback.
        Returns:
            The EarlyStopping callback.
    """
    return EarlyStopping(
    monitor='loss',  # (ex. loss or accuracy)
    patience=10,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)




# -------------------------------------------- SAVE MODEL -------------------------------------------- #

def save_model(model: tf.keras.Model, name: str = HRAS_MODEL_NAME):
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



# -------------------------------------------- PREDICT -------------------------------------------- #


def model_predict(model: tf.keras.Model, X: pd.DataFrame, pfam = False, 
                  use_P53_v2 = True) -> tuple[tf.Tensor, np.ndarray]:
    """
        Predict the labels for the given features.
        Parameters:
            model: The trained model.
            X: The features to predict.
            pfam: If True, the model will be trained with Pfam data. Default is False.
            use_P53_v2: If True, the model is trained from P53 V2 Model. Default is False.
        Returns:
            Predicted probabilities: as a Tensor
            Predicted labels: as a numpy array of indexes
    """
    # Convert to Tensor
    #X_tensor = tf.convert_to_tensor(X.values, dtype=tf.float32)

    # Predict Probabilities
    if use_P53_v2:
        probabilities = model.predict(input_dict_prepare(X, pfam))
    else:
        probabilities = model.predict(X)

    return probabilities, tf.argmax(probabilities, axis=1).numpy()