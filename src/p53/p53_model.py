"""
    This module contains the p53 MLP model code.
    6. This module should be the sixth module in the pipeline.
"""

import os
import tensorflow as tf
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, Dropout
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Input
from src import P53_MODEL_DIR, P53_MODEL_NAME

def p53_train_model(X_train, y_train, X_test, y_test) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
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
        Returns:
            The trained model and the training history.
    """

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.3),
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
    print("\nTest Evaluation:")
    print(model.evaluate(X_test, tf.keras.utils.to_categorical(y_test)))

    return model, history




def save_model(model: tf.keras.Model, name: str = P53_MODEL_NAME):
    """
        Save the model to a file. 
        If a model with the same name exists, it will be renamed with a .bak extension.
        Files are saved in the models directory.
        Models are saved in both .h5 and .keras formats.

        Parameters:
            model: The model to save.
            name: The name of the model
    """

    model_path = f"{P53_MODEL_DIR}/{name}.h5"
    keras_path = f"{P53_MODEL_DIR}/{name}.keras"

    try:
        if os.path.exists(P53_MODEL_DIR) is False:
            os.makedirs(P53_MODEL_DIR)

        elif os.path.exists(model_path):
            back_path = model_path + ".bak"
            if os.path.exists(back_path):
                os.remove(back_path)
            os.rename(src=model_path, dst= model_path + ".bak")

        if os.path.exists(keras_path):
            back_path = keras_path + ".bak"
            if os.path.exists(back_path):
                os.remove(back_path)
            os.rename(src=keras_path, dst=keras_path + ".bak")      

        model.save(model_path)
        model.save(keras_path)

    except Exception as e:
        print(f"Error saving model: {e}")
        print("Trying to save the model with alternative name: `alt.h5`...")
        model.save(f"{P53_MODEL_DIR}/alt.h5")
        return
    



def train_model_to_save(model, X_train, y_train) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
        Train a model with all the given data in order to save it.
        It does not split the data into training and test sets.
        It doesn't save the model.
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