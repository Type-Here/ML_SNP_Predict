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