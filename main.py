import os
import tensorflow as tf
import matplotlib.pyplot as plt

from src.dataset_file_management import load_data
from src.config import MODELS_DIR, P53_MODEL_NAME

import src.p53.p53_data_prep as p53_data_prep
import src.p53.p53_encoding as p53_encoding
import src.p53.p53_scaling as p53_scaling
import src.p53.p53_data_balancing as p53_data_balancing
import src.p53.p53_train_test_sets as p53_split
import src.p53.p53_model as p53_model
import src.model_evaluation as ev

def main():
    # If a p53 model exists, load it
    if os.path.exists(f"{MODELS_DIR}/{P53_MODEL_NAME}.keras"):
        print(" -- Model found --\n")
        print("Loading model...")
        model = tf.keras.models.load_model(f"{MODELS_DIR}/{P53_MODEL_NAME}.keras")
        model.trainable = False
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Otherwise, train a new model
    else:
        print("-- Model not found --\n")
        print("Training a new model...")
        data = load_data(protein = 'p53')
        print(data.head())

        # Preprocess the data
        print("Preprocessing the data...")
        
        print("Cleaning the data...")
        data = p53_data_prep.p53_cleaning(data)
        
        print("Encoding the data...")
        data = p53_encoding.p53_encoding(data)
        
        print("Scaling the data...")
        data = p53_scaling.p53_scaling(data)

        print("Balancing the data...")
        X_resampled, y_resampled = p53_data_balancing.balance_split_data(data)

        print("Creating the training and test sets...")
        X_train, X_test, y_train, y_test \
            = p53_split.create_train_test_sets(X_data=X_resampled, y_labels=y_resampled)
        
        print("Training the model...")
        model, history = p53_model.p53_train_model(X_train, y_train, X_test, y_test)

        print("Evaluating the model...")

        # print("\nModel history - Accuracy plot:")
        # ev.plot_accuracy(history) # Unable to plot in the current environment

        print("\nModel evaluation:")
        ev.simple_evaluate_model(model, X_test, y_test)

        print("\nCross-validation evaluation:")
        ev.n_times_k_fold_eval(model, X_resampled, y_resampled, n_splits=10, n_repeats=1)

        model, history = p53_model.train_model_to_save(model, X_resampled, y_resampled)

        print("Saving the model...")
        p53_model.save_model(model)

        print("Model saved.")

        # print("\nPlotting the accuracy...")
        # ev.plot_curve_auc(model, X_test, y_test) # Unable to plot in the current environment

        
    # If the model was trained or loaded, print the model summary
    print("Model summary:")
    print(model.summary())

    if 'history' in locals():
        print("History:")
        print(history.history)

    print("Done.")




# Run the main function
if __name__ == "__main__":
    main()
