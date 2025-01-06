"""
    This module contains the functions to evaluate the p53 model.
    7. This module should be the seventh module in the pipeline.
"""
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import json
import os

from src.config import MODELS_STATS_DIR, PLOTS_SAVE_DIR
from src.p53.p53_6_model_v2 import input_dict_prepare


def simple_evaluate_model(model, X_test, y_test):
    """
        Evaluate the model using the test set.
        Print the accuracy, classification report, and confusion matrix.
    """
    # Get the predicted classes and probabilities
    y_pred, y_pred_probs, y_test_classes = _get_classes(model, X_test, y_test)

    # Accuracy
    accuracy = accuracy_score(y_test_classes, y_pred)
    print(f"\nAccuracy: {accuracy:.2f}")

    # Classification report (Precision, Recall, F1-Score)
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_classes, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)




def n_times_k_fold_eval(model, X_resampled, y_resampled, n_splits=5, n_repeats=5, pfam=False):
    """
        Perform n-times k-fold cross-validation on the model.

        Args:
            - model: The model to evaluate.
            - X_resampled: The resampled features. All features after encoding and balancing without Labels.
            - y_resampled: The resampled labels. All labels after encoding and balancing.
            - n_splits: The number of splits for the KFold. Default is 5. K value for KFold.
            - n_repeats: The number of times to repeat the KFold. Default is 5. N times to repeat KFold.
            - pfam: Whether to use the Pfam data. Default is False.
    """

    # Cumulative metrics
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # KFold per k-fold validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for repeat in range(n_repeats):
        print(f"\nRepeat {repeat + 1}/{n_repeats}")
        for train_index, test_index in kf.split(X_resampled):
            # Split dataset
            X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
            y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

            # Prepare the data
            X_train = input_dict_prepare(X_train, pfam)
            X_test = input_dict_prepare(X_test, pfam)

            # Convert labels to one-hot encoding for training
            y_train_onehot = tf.keras.utils.to_categorical(y_train)
            y_test_onehot = tf.keras.utils.to_categorical(y_test)

            # Train the model
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train_onehot, epochs=50, batch_size=32, verbose=0)

            # Evaluate on the test set
            y_pred_probs = model.predict(X_test)
            y_pred = y_pred_probs.argmax(axis=1)

            # Calculate metrics for this fold
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
            recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    # Final evaluation
    print("\nFinal Evaluation over 10-times 10-fold:")
    print(f"Accuracy: {np.mean(accuracy_scores):.3f} ± {np.std(accuracy_scores):.3f}")
    print(f"Precision: {np.mean(precision_scores):.3f} ± {np.std(precision_scores):.3f}")
    print(f"Recall: {np.mean(recall_scores):.3f} ± {np.std(recall_scores):.3f}")
    print(f"F1-Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")



def plot_accuracy(history, model_name = None):
    """
    Plot the accuracy of the model during training.
    
    Parameters:
        history: The training history object returned by Keras model.fit.
        model_name: The name of the model to display in the plot title. Default is None: generic title.
    Returns:
        matplotlib.figure.Figure: The generated accuracy plot.
    """
    figure, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history.history['accuracy'], label='Accuracy', linewidth=2)
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    if model_name:
        title = model_name.replace("_", " ").title()
        ax.set_title(f'{title}', fontsize=16)
    else:
        ax.set_title('Model Accuracy', fontsize=16)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return figure


def plot_loss(history, model_name = None):
    """
    Plot the loss of the model during training.
    
    Parameters:
        history: The training history object returned by Keras model.fit.
        model_name: The name of the model to display in the plot title. Default is None: generic title.
    Returns:
        matplotlib.figure.Figure: The generated loss plot.
    """
    figure, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history.history['loss'], label='Loss', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    if model_name:
        title = model_name.replace("_", " ").title()
        ax.set_title(f'{title}', fontsize=16)
    else:
        ax.set_title('Model Loss', fontsize=16)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return figure



def plot_curve_auc(model, X_test, y_test, model_name=None):
    """
    Plot the ROC Curve and compute AUC for each class.
    
    Parameters:
        model: The trained model.
        X_test: Test data.
        y_test: True labels for test data.
        model_name: The name of the model to display in the plot title. Default is None: generic title.
        
    Returns:
        matplotlib.figure.Figure: The generated ROC curve plot.
    """
    fpr = {}
    tpr = {}
    roc_auc = {}

    y_pred, y_pred_probs, y_test_classes = _get_classes(model, X_test, y_test)

    for i in range(len(np.unique(y_test_classes))):  # Number of classes
        fpr[i], tpr[i], _ = roc_curve(y_test_classes == i, y_pred_probs[:, i])
        roc_auc[i] = roc_auc_score(y_test_classes == i, y_pred_probs[:, i])

    figure, ax = plt.subplots(figsize=(10, 8))
    for i in range(len(fpr)):
        ax.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')  # Random classifier line
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    if model_name:
        title = model_name.replace("_", " ").title()
        ax.set_title(f'{title}', fontsize=16)
    else:
        ax.set_title("ROC Curve", fontsize=16)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return figure


# -------------------- Helper functions -------------------- #

def _get_classes(model, X_test, y_test):
    """
        Get the predicted classes and probabilities.
    """
    # Predictions
    y_pred_probs = model.predict(X_test)

    # Convert probabilities to classes
    if y_pred_probs.shape[1] > 1:  # Multiclass
        y_pred = y_pred_probs.argmax(axis=1)
    else:  # if binary
        y_pred = (y_pred_probs > 0.5).astype(int)

    # Manage y_test_classes
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:  # One-hot encoding
        y_test_classes = y_test.argmax(axis=1)
    else:  # if already numerical
        y_test_classes = y_test
    
    return y_pred, y_pred_probs, y_test_classes


# -------------------- Save statistics -------------------- #


def save_extended_stats(model, X_test, y_test, model_name, history: keras.callbacks.History):
    """
    Save the extended evaluation statistics of the model, including accuracy, recall, specificity,
    precision, F1-score, loss, and data for AUC and loss plots.

    Parameters:
        model (tf.keras.Model): Keras model to evaluate.
        X_test (np.ndarray): Test dataset.
        y_test (np.ndarray): Test labels.
        model_name (str): Name of the model.
        history (tf.keras.callbacks.History): History object from model training.
    """
    # Calculate predicted classes and probabilities
    y_pred, y_pred_probs, y_test_classes = _get_classes(model, X_test, y_test)

    # Accuracy
    accuracy = accuracy_score(y_test_classes, y_pred)

    # Precision, Recall, F1-Score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_classes, y_pred, average=None)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_classes, y_pred)

    # Specificity (1 - False Positive Rate)
    specificity = []
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity.append(tn / (tn + fp))

    # Loss data
    loss_data = history.history.get('loss', [])
    val_loss_data = history.history.get('val_loss', [])

    # Data for ROC/AUC
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(len(np.unique(y_test_classes))):
        fpr[i], tpr[i], _ = roc_curve(y_test_classes == i, y_pred_probs[:, i])
        roc_auc[i] = roc_auc_score(y_test_classes == i, y_pred_probs[:, i])

    # Data structure to save
    stats = {
        "accuracy": accuracy,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "specificity_per_class": specificity,
        "loss_data": loss_data,
        "val_loss_data": val_loss_data,
        "roc_auc": {str(k): v for k, v in roc_auc.items()},
        "fpr_tpr": {str(k): {"fpr": fpr[k].tolist(), "tpr": tpr[k].tolist()} for k in fpr}
    }

    # Save in JSON format
    save_path = f"{MODELS_STATS_DIR}/{model_name}_stats.json"
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"Statistiche estese salvate in: {save_path}")



# -------------------- Load statistics -------------------- #

import json

def load_stats(model_name):
    """
    Reads the statistics data from a JSON file and returns it as a dictionary.

    Parameters:
        model_name (str): Name of the model to identify the stats file.

    Returns:
        dict: Data loaded from the JSON file.
    """
    # Construct the file path for the stats JSON
    file_path = f"{MODELS_STATS_DIR}/{model_name}_stats.json"
    
    try:
        # Open and read the JSON file
        with open(file_path, 'r') as f:
            stats = json.load(f)
        print(f"Data loaded from: {file_path}")
        return stats
    except FileNotFoundError:
        # Handle the case where the file does not exist
        print(f"Error: The file {file_path} does not exist.")
        return None
    except json.JSONDecodeError as e:
        # Handle the case where the JSON is invalid
        print(f"Error decoding the JSON file: {e}")
        return None



# --------------------  Save Plot -------------------- #



def save_plot_incremental(figure, model_name, output_dir=PLOTS_SAVE_DIR):
    """
    Saves a plot incrementally by appending a number to the file name if it already exists.
    
    Parameters:
        figure (matplotlib.figure.Figure): The figure to save.
        base_name (str): The base name for the file (e.g., "auc", "loss", "accuracy").
        output_dir (str): Directory where the plots will be saved. Default is 'plots'.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the initial file path
    file_path = os.path.join(output_dir, f"{model_name}.svg")
    counter = 1
    
    # Increment file name if it already exists
    while os.path.exists(file_path):
        file_path = os.path.join(output_dir, f"{model_name}_{counter}.svg")
        counter += 1
    
    # Save the figure
    figure.savefig(file_path, bbox_inches="tight", format="svg")
    plt.close(figure)  # Close the figure to free memory
    print(f"Plot saved as: {file_path}")

# Example usage:
# Assuming you have matplotlib figures for AUC, Loss, and Accuracy:
# save_plot_incremental(auc_figure, "auc")
# save_plot_incremental(loss_figure, "loss")
# save_plot_incremental(accuracy_figure, "accuracy")
