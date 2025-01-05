import sys
import traceback
from PyQt5.QtCore import QThread, pyqtSignal

from UI.stream.stream_redirect import StreamRedirector

from src.dataset_file_management import load_data, save_processed_data
from src.config import P53_MODEL_NAME

import src.p53.p53_1_data_prep as p53_1_data_prep
import src.p53.p53_2_encoding as p53_2_encoding
import src.p53.p53_3_scaling as p53_3_scaling
import src.p53.p53_4_data_balancing as p53_4_data_balancing
import src.p53.p53_5_data_split_train as p53_split
import src.p53.p53_6_model as p53_6_model
import src.model_evaluation as ev

class TrainingThread(QThread):
    log_signal = pyqtSignal(str)  # Signal to send log messages to the main thread (GUI)

    def run(self):
        """
            Run the training process in a separate thread.
        """
        
        # Redirect stdout and stderr to capture Keras output
        stream_redirector = StreamRedirector()
        stream_redirector.log_signal.connect(self.log_signal)  # Connect to main thread
        sys.stdout = stream_redirector
        sys.stderr = stream_redirector


        try:            
            self.log_signal.emit("Starting the training process...\n")

            data = load_data(protein = 'p53') # Loads saved data or downloads it if not present
            if data is None:
                self.log_signal.emit("Error loading data.")
                return
            self.log_signal.emit("Dataset loaded.")
            self.log_signal.emit(f"Data: {data.shape[0]} x {data.shape[1]}\n")

            # Preprocess the data
            self.log_signal.emit("Preprocessing the data...")
            
            self.log_signal.emit("Cleaning the data...")
            data = p53_1_data_prep.p53_cleaning(data)

            data_save_path = save_processed_data(data, P53_MODEL_NAME) # Save the processed data Not Encoded and Scaled
            if data_save_path:
                self.log_signal.emit(f"Processed non encoded dataset saved to:  \n{data_save_path}\n")
            else:
                self.log_signal.emit("Error saving processed non encoded dataset.")
            
            self.log_signal.emit("Encoding the data...")
            data = p53_2_encoding.p53_encoding(data)
            
            self.log_signal.emit("Scaling the data...")
            data = p53_3_scaling.p53_scaling(data)

            data_save_path = save_processed_data(data, P53_MODEL_NAME, is_encoded=True, ) # Save the processed data Encoded and Scaled but not Balanced
            if data_save_path:
                self.log_signal.emit(f"Processed Encoded dataset saved to:  \n{data_save_path}\n")
            else:
                self.log_signal.emit("Error saving processed Encoded dataset.")

            self.log_signal.emit("Balancing the data...")
            X_resampled, y_resampled = p53_4_data_balancing.balance_split_data(data)

            self.log_signal.emit("Creating the training and test sets...")
            X_train, X_test, y_train, y_test \
                = p53_split.create_train_test_sets(X_data=X_resampled, y_labels=y_resampled)
            
            # Train the model
            self.log_signal.emit("Training the model...")
            model, history = p53_6_model.p53_train_model(X_train, y_train, X_test, y_test)

            # Evaluate the model
            self.log_signal.emit("Evaluating the model...")

            # self.log_signal.emit("\nModel history - Accuracy plot:")
            # ev.plot_accuracy(history) # Unable to plot in the current environment

            self.log_signal.emit("\nModel evaluation:")
            ev.simple_evaluate_model(model, X_test, y_test)

            self.log_signal.emit("\nSaving the model statistics...")
            ev.save_extended_stats(
                model=model, 
                X_test=X_test, 
                y_test=y_test,
                history=history,
                model_name=P53_MODEL_NAME
            )

            self.log_signal.emit("\nCross-validation evaluation:")
            ev.n_times_k_fold_eval(model, X_resampled, y_resampled, n_splits=10, n_repeats=2)

            self.log_signal.emit("Training complete!\n ------------------- \n")

            # Train the model to save it
            self.log_signal.emit("Training the model with all the data to save it...")

            model, history = p53_6_model.retrain_model_to_save(model, X_resampled, y_resampled)

            # Save the model
            self.log_signal.emit("\nSaving the model...")

            p53_6_model.save_model(model)

            self.log_signal.emit("Model saved.")

            self.log_signal.emit("Training complete!\n")

            self.log_signal.emit(" -- Done. --")

            self.log_signal.emit("Reload the model to make predictions and see statistics.")

        except Exception as e:
            self.log_signal.emit(f"Error during training: {e}")
            self.log_signal.emit(f"Traceback: {traceback.format_exc()}")
            self.log_signal.emit("Error: Training failed.")
            self.log_signal.emit(" -- Done. --")
            return
        
        finally:
            # Restore stdout and stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__