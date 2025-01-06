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
from keras.api.models import Sequential, Model
from keras.api.layers import Dense, Dropout, Input, Embedding, Flatten, Concatenate
from keras.api.callbacks import EarlyStopping
from src import MODELS_DIR, P53_MODEL_NAME
from src.models_usage import save_model as general_save_model

# -------------------------------------------- CONSTANTS -------------------------------------------- #

N5_columns = ['WT_Codon_First', 'WT_Codon_Second', 'WT_Codon_Third',
                        'Mutant_Codon_First', 'Mutant_Codon_Second', 'Mutant_Codon_Third']
N4_columns = ['cDNA_Ref', 'cDNA_Mut']
AA21_columns = ['WT AA_1','Mutant AA_1']

# -------------------------------------------- TRAIN MODEL -------------------------------------------- #

def p53_train_model(X_train, y_train, X_test, y_test, pfam=False) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Create and train the p53 MLP model.
    Model architecture:
        - Embedding layer for each of the columns
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
    # Create input layers
    input_layers, concat = __add_input_layer(pfam)
    
    # Define dense layers
    x = Dense(128, activation='relu')(concat)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(3, activation='softmax')(x)

    # Build the model
    model = Model(inputs=input_layers, outputs=output)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Prepare the data dictionary
    X_train_dict = input_dict_prepare(X_train, pfam)
    X_test_dict = input_dict_prepare(X_test, pfam)

    # Train the model
    history = model.fit(
        x=X_train_dict,
        y=tf.keras.utils.to_categorical(y_train),
        validation_data=(X_test_dict, tf.keras.utils.to_categorical(y_test)),
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[__early_stopping()]
    )

    return model, history



# -------------------------------------------- AIDE FUNCTIONS -------------------------------------------- #


def __add_input_layer(pfam:bool) -> tf.keras.layers.Layer:
    """
        Add Input Layer for Embeddings to the model.
        Parameters:
            model: The model to add the input layer to.
            dataset: The dataset to get the input shape from.
            pfam: If True, the model will be trained with Pfam data. Default is False.
        Returns:
            The model with the input layer added.
    """
    input_layers = []

    # Input layers for encoded columns
    for column in N5_columns + N4_columns + AA21_columns:
        input_layers.append(Input(shape=(1,), name=column + '_Encoded'))

    # Embedding layers
    embedding_layers = []
    for i, column in enumerate(N5_columns):
        embedding = Embedding(input_dim=5, output_dim=2)(input_layers[i])
        embedding_layers.append(Flatten()(embedding))

    for i, column in enumerate(N4_columns):
        embedding = Embedding(input_dim=4, output_dim=2)(input_layers[len(N5_columns) + i])
        embedding_layers.append(Flatten()(embedding))

    for i, column in enumerate(AA21_columns):
        embedding = Embedding(input_dim=21, output_dim=5)(input_layers[len(N5_columns) + len(N4_columns) + i])
        embedding_layers.append(Flatten()(embedding))

    # Numerical inputs
    if pfam:
        numerical_input = Input(shape=(2,), name='Position and Conservation')
    else:
        numerical_input = Input(shape=(1,), name='cDNA_Position')
    input_layers.append(numerical_input)
    embedding_layers.append(numerical_input)

    # Concatenate embeddings and numerical input
    concat = Concatenate()(embedding_layers)
    return input_layers, concat


def __early_stopping():
    """
        Create an EarlyStopping callback.
        Returns:
            The EarlyStopping callback.
    """
    return EarlyStopping(
    monitor='val_loss',  # (ex. val_loss or val_accuracy)
    patience=10,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)


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



# -------------------------------------------- SAVE MODEL -------------------------------------------- #


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


# -------------------------------------------- RETRAIN MODEL FOR SAVE -------------------------------------------- #


def retrain_model_to_save(model, X_train, y_train, pfam = False) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
        Train a model with all the given data in order to save it.
        It does not split the data into training and test sets.
        It doesn't save the model.
        Parameters:
            model: The model to retrain.
            X_train: The features to train the model.
            y_train: The labels to train the model.
            pfam: If True, the model will be trained with Pfam data. Default is False.
        Returns:
            The trained model and the training history.
    """

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        x = input_dict_prepare(X_train,pfam), 
        y = tf.keras.utils.to_categorical(y_train), 
        epochs=50, 
        batch_size=32, 
        verbose=1,
        callbacks=[__early_stopping()]
    )

    return model, history



# -------------------------------------------- PREDICT -------------------------------------------- #


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
    #X_tensor = tf.convert_to_tensor(X.values, dtype=tf.float32)

    # Predict Probabilities
    probabilities = model.predict(input_dict_prepare(X))

    return probabilities, tf.argmax(probabilities, axis=1).numpy()