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

from src.config import MODELS_STATS_DIR


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




def n_times_k_fold_eval(model, X_resampled, y_resampled, n_splits=5, n_repeats=5):
    """
        Perform n-times k-fold cross-validation on the model.

        Args:
            - model: The model to evaluate.
            - X_resampled: The resampled features. All features after encoding and balancing without Labels.
            - y_resampled: The resampled labels. All labels after encoding and balancing.
            - n_splits: The number of splits for the KFold. Default is 5. K value for KFold.
            - n_repeats: The number of times to repeat the KFold. Default is 5. N times to repeat KFold.
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



def plot_accuracy(history):
    """
        Plot the accuracy of the model during training.
    """
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_loss(history):
    """
        Plot the loss of the model during training.
    """
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_curve_auc(model, X_test, y_test):
    # Compute ROC Curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    y_pred, y_pred_probs, y_test_classes = _get_classes(model, X_test, y_test)

    for i in range(len(np.unique(y_test_classes))):  # Number of classes
        fpr[i], tpr[i], _ = roc_curve(y_test_classes == i, y_pred_probs[:, i])
        roc_auc[i] = roc_auc_score(y_test_classes == i, y_pred_probs[:, i])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(len(fpr)):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


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
