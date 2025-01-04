"""
    This module contains functions to load models.
"""

import tensorflow as tf
import os
from src import MODELS_DIR, P53_MODEL_NAME, P53_PFAM_MODEL_NAME, HRAS_MODEL_NAME

## --------------------------- LOAD MODELS --------------------------- ##

def load_model_by_name(name: str) -> tf.keras.Model:
    """
        Load a model by common name.
        Parameters:
            name: The name of the model to load from GUI or CLI. (Model name seen by the user)
        Returns:
            tf.keras.Model: The loaded model.
    """
    model_name = None
    match name:
        case "P53 Model":
            model_name = P53_MODEL_NAME
        case "P53 Pfam":
            model_name = P53_PFAM_MODEL_NAME
        case "Hras Transfer":
            model_name = HRAS_MODEL_NAME
        case _:
            return None
    
    if os.path.exists(f"{MODELS_DIR}/{model_name}.keras"):
        model = tf.keras.models.load_model(f"{MODELS_DIR}/{model_name}.keras")
        model.trainable = False
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
    else:
        return None
    

## --------------------------- SAVE MODELS --------------------------- ##

def save_model(model: tf.keras.Model, name: str):
    """
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

    model_path = f"{MODELS_DIR}/{name}.h5"
    keras_path = f"{MODELS_DIR}/{name}.keras"

    try:
        if os.path.exists(MODELS_DIR) is False:
            os.makedirs(MODELS_DIR)

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
        return keras_path
    
    except Exception as e:
        print(f"Error saving model: {e}")
        print("Trying to save the model with alternative name: `alt.h5`...")
        model.save(f"{MODELS_DIR}/alt_{name}.h5")
        return None
    



## --------------------------- PREDICTIONS --------------------------- ##

def get_prediction(model_name, model, position, ref, mut, sequence):
    """
        Get the prediction of a model.
        Parameters:
            model_name: The name of the model as seen by the user.
            model: The model to use for the prediction.
            position: The position of the mutation.
            ref: The reference amino acid.
            mut: The mutated amino acid.
            sequence: The protein sequence.
        Returns:
            The prediction of the model.
    """

    match model_name:
        case "P53 Model":
            return _p53_prediction(model, position, ref, mut, sequence)
        case "P53 Pfam":
            return _p53_prediction(model, position, ref, mut, sequence, pfam=True)
        case "Hras Transfer":
            return _hras_prediction(model, position, ref, mut, sequence)
        case _:
            return None
        

def _p53_prediction(model, position, ref, mut, sequence, pfam=False):
    """
        Get the prediction of the p53 model.
        Parameters:
            model: The model to use for the prediction.
            position: The position of the mutation.
            ref: The reference amino acid.
            mut: The mutated amino acid.
            sequence: The protein sequence.
            pfam: Whether to use the Pfam model. Default is False.
        Returns:
            The prediction of the model.
    """
    if pfam:
        return None
    else:
        return None
    



def _hras_prediction(model, position, ref, mut, sequence):
    """
        Get the prediction of the hras model.
        Parameters:
            model: The model to use for the prediction.
            position: The position of the mutation.
            ref: The reference amino acid.
            mut: The mutated amino acid.
            sequence: The protein sequence.
        Returns:
            The prediction of the model.
    """
    return None