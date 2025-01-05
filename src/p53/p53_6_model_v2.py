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

def p53_train_model(X_train, y_train, X_test, y_test) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
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
        Returns:
            The trained model and the training history.
    """
    input_layer = __add_input_layer()
    model = Sequential([
        input_layer,
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

     # Prepara il dizionario dei dati
    X_train_dict = __input_dict_prepare(X_train)
    X_test_dict = __input_dict_prepare(X_test)


    # Train the model
    history = model.fit(
        x=X_train_dict,
        y=tf.keras.utils.to_categorical(y_train),  # Convert labels to one-hot
        validation_data=(X_test_dict, tf.keras.utils.to_categorical(y_test)),
        epochs=50,  # Adjusted based on performance
        batch_size=32,
        verbose=1,
        callbacks=[__early_stopping()]
    )

    # Evaluate the model
    #print("\nTest Evaluation:")
    #print(model.evaluate(X_test, tf.keras.utils.to_categorical(y_test)))

    return model, history


# -------------------------------------------- AIDE FUNCTIONS -------------------------------------------- #


def __add_input_layer() -> tf.keras.layers.Layer:
    """
        Add Input Layer for Embeddings to the model.
        Parameters:
            model: The model to add the input layer to.
            dataset: The dataset to get the input shape from.
        Returns:
            The model with the input layer added.
    """

    input_layers = []

    # Input layers 
    for column in N5_columns:
        input_layers.append(Input(shape=(1,), name=column + '_Encoded'))
    for column in N4_columns:
        input_layers.append(Input(shape=(1,), name=column + '_Encoded'))
    for column in AA21_columns:
        input_layers.append(Input(shape=(1,), name=column + '_Encoded'))

    # Embedding layers
    embedding_layers = []
    
    for i, column in enumerate(N5_columns):
        embedding = Embedding(input_dim=5, output_dim=2)(input_layers[i])
        flat = Flatten()(embedding)
        embedding_layers.append(flat)
    
    for i, column in enumerate(N4_columns):
        embedding = Embedding(input_dim=4, output_dim=2)(input_layers[len(N5_columns) + i])
        
        if column == 'cDNA_Mut':  # Add Weight to cDNA_Mut
            weighted_flat = tf.keras.layers.Lambda(lambda x: x * 2.0)(flat)  # NOTE: This weight can be adjusted
            embedding_layers.append(weighted_flat)
        else:
            embedding_layers.append(Flatten()(embedding))
    
    for i, column in enumerate(AA21_columns):
        embedding = Embedding(input_dim=21, output_dim=5)(input_layers[len(N5_columns) + len(N4_columns) + i])
        embedding_layers.append(Flatten()(embedding))


    
    # Numerical inputs
    numerical_input = Input(shape=(1,), name='cDNA_Position')  # 2 Numerical inputs: cDNA_Position
    embedding_layers.append(numerical_input)

    # Concatenate embeddings and numerical input
    concat = Concatenate()(embedding_layers)
    return concat


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


def __input_dict_prepare(X: pd.DataFrame) -> dict:
    """
        Prepare the dictionary of data for the model.
        Parameters:
            X: The features to predict.
        Returns:
            dict: The dictionary of data for the model.
    """
    X_dict = {column + '_Encoded': X[column + '_Encoded'] \
                    for column in N5_columns + N4_columns + AA21_columns}
    X_dict['cDNA_Position'] = X['cDNA_Position']
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


def retrain_model_to_save(model, X_train, y_train) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
        Train a model with all the given data in order to save it.
        It does not split the data into training and test sets.
        It doesn't save the model.
        Returns:
            The trained model and the training history.
    """

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        x = __input_dict_prepare(X_train), 
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
    probabilities = model.predict(__input_dict_prepare(X))

    return probabilities, tf.argmax(probabilities, axis=1).numpy()