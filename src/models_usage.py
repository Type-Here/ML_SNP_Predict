"""
    This module contains functions to load models.
"""

import tensorflow as tf
import os
from src import MODELS_DIR, P53_MODEL_NAME, P53_PFAM_MODEL_NAME, HRAS_MODEL_NAME


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