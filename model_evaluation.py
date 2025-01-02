"""
    This module contains the functions to evaluate the p53 model.
    7. This module should be the seventh module in the pipeline.
"""
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow as tf


import numpy as np

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