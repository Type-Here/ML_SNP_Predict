"""
    This module contains functions to load models.
"""

import traceback
import tensorflow as tf
import pandas as pd
import os
from src import MODELS_DIR, P53_MODEL_NAME, P53_PFAM_MODEL_NAME, HRAS_MODEL_NAME
from src.dataset_file_management import load_codons_aa_json, load_processed_data
from src.p53.p53_2_encoding import __one_hot_encoding, p53_encoding

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

pathogenicity_labels = {
    0: "Benign",
    1: "Pathogenic",
    2: "Uncertain"
}

def get_prediction(model_name:str, model:tf.keras.Model,
                    position:str, ref:str, mut:str, sequence:str):
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
    from src.p53.p53_6_model import model_predict as p53_predict # Import here to avoid circular import
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
            The probability and label of the prediction. If an error occurs, None is returned.
    """
    if pfam:
        return None
    else:
        try:
            # Load codons to amino acids mapping JSON file
            codons_aa = load_codons_aa_json()

            # Load processed dataset
            processed_data = load_processed_data(P53_MODEL_NAME)
            if processed_data is None:
                return None
            
            position = int(position)
            
            # Get the codon from the sequence
            start = position - (position % 3)
            end = start + 3
            codon = sequence[start:end]
            
            # Get the mutated codon
            mut_codon = list(codon)
            mut_codon[position % 3] = mut  # Replace the reference amino acid with the mutated amino acid
            mut_codon = ''.join(mut_codon)

            # Get the amino acids
            wt_aa = __get_codon_aa_mapping(codons_aa, codon)
            mut_aa = __get_codon_aa_mapping(codons_aa, mut_codon)
            print(f"WT: {wt_aa} -> Mut: {mut_aa}")

            input={}
            input['cDNA_position'] = position
            input['WT_Codon_First'] = codon[0]
            input['WT_Codon_Second'] = codon[1]
            input['WT_Codon_Third'] = codon[2]

            input['Mut_Codon_First'] = mut_codon[0]
            input['Mut_Codon_Second'] = mut_codon[1]
            input['Mut_Codon_Third'] = mut_codon[2]

            input['cDNA_ref'] = ref
            input['cDNA_mut'] = mut

            input['WT_AA'] = wt_aa
            input['Mut_AA'] = mut_aa

            input = __get_domain_mapping(input, position, processed_data)

            # Convert to DataFrame
            pd_input = pd.DataFrame(input, index=[0])

            encoded = p53_encoding(pd_input, isPrediction=True)
            encoded = __one_hot_encode_domain(encoded, __get_domains(processed_data))

            # Get the prediction
            prob, prediction = p53_predict(model, encoded)
            print(f"Prediction: {prob}, Label: {pathogenicity_labels[prediction]}")

            return prob, pathogenicity_labels[prediction]

            
        except FileNotFoundError:
            print("Error loading codons to amino acids JSON file.")
            return None
        except Exception as e:
            print(f"Error getting prediction: {e} \n")
            traceback.print_exc()  # Print the traceback
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



# --------------------------- CODON-AMINOACID MAPPING --------------------------- #

def __get_codon_aa_mapping(map, codon):
    """
        Get the amino acid corresponding to a codon.
        Parameters:
            map: The codon to amino acid mapping.
            codon: The codon to get the amino acid for.
        Returns:
            The amino acid corresponding to the codon. If the codon is not found, '0' is returned.
    """
    return map.get(codon, '0')


# --------------------------- P53_MODEL DOMAIN MAPPING --------------------------- #

def __get_domain_mapping(input, position, processed_data: pd.DataFrame):
    """
        Get the domain of the codon reference amino acid for the p53 model.
        Parameters:
            position: The position of the mutation.
        Returns:
            The domain of the p53 model.
    """
    
    # Get the domain of the codon reference amino acid
    # by finding the closest domain to the position in the processed data

    domain = processed_data.loc[processed_data['cDNA_Position'] <= position].iloc[-1]['Domain']
    input['Domain'] = domain

    return input

def __get_domains(processed_data: pd.DataFrame):
    """
        Get the list of domains from the processed data.
        Parameters:
            processed_data: The processed data.
        Returns:
            The list of domains.
    """
    return processed_data['Domain'].unique()


def __one_hot_encode_domain(input, domains):
    """
        One-hot encode the domain.
        Parameters:
            input: The input data.
            domains: The domains list.
        Returns:
            The input data with the domain one-hot encoded.
    """
    input = __one_hot_encoding(input, ['Domain'], domains)
    return input