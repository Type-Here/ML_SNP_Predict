import sys
import requests
import traceback
from PyQt5.QtCore import QThread, pyqtSignal

from UI.stream.stream_redirect import StreamRedirector

from src.dataset_file_management import load_data, save_processed_data
from src.pfam.pfam_json import get_pfam_data, create_domain_dict, assign_conservation
from src.config import P53_MODEL_NAME, P53_PFAM_MODEL_NAME, HRAS_MODEL_NAME

import src.model_evaluation as ev
import src.models_usage as mu

class TrainingThread(QThread):
    log_signal = pyqtSignal(str)  # Signal to send log messages to the main thread (GUI)


    def __init__(self, model_name, set_pfam = False, set_email = "fia@unisa.it", parent=None):
        super().__init__(parent)
        self.use_pfam = set_pfam
        self.model_name = model_name
        self.email = set_email

    
    def set_use_pfam(self, set_pfam: bool):
        self.use_pfam = set_pfam
        self.log_signal.emit(f"Use Pfam: {self.use_pfam}")

    def set_email(self, set_email: str):
        self.email = set_email
        self.log_signal.emit(f"Email: {self.email}")


    def run(self):
        """
            Run the training process in a separate thread.
        """
        try:
            if self.model_name == HRAS_MODEL_NAME:
                self.__train_hras_model()
                return

            elif self.model_name == P53_MODEL_NAME:
                self.__train_p53_model()

            elif self.model_name == P53_PFAM_MODEL_NAME:
                if not self.use_pfam:
                    self.log_signal.emit("Error: Pfam data is set to False.")
                    self.log_signal.emit("To train the Pfam model, set Pfam data to True.")
                    self.log_signal.emit("Error: Training failed.")
                    return
                self.__train_p53_model()
                    
            else:
                self.log_signal.emit(f"Error: Model name not recognized: {self.model_name}")
                self.log_signal.emit("Error: Training failed.")
        except requests.exceptions.RequestException as e:
            self.log_signal.emit(f"Error during training: Internet connection required." + \
                                   "Check your connection.")

    # -------------------------------------------- P53 MODEL -------------------------------------------- #


    def __train_p53_model(self):
        """
            Train the p53 model.
        """
        # Local imports to avoid circular imports
        import src.p53.p53_1_data_prep as p53_1_data_prep
        import src.p53.p53_2_encoding as p53_2_encoding
        import src.p53.p53_3_scaling as p53_3_scaling
        import src.p53.p53_4_data_balancing as p53_4_data_balancing
        import src.p53.p53_5_data_split_train as p53_split
        import src.p53.p53_6_model_v2 as p53_6_model


        # Redirect stdout and stderr to capture Keras output
        stream_redirector = StreamRedirector()
        stream_redirector.log_signal.connect(self.log_signal)  # Connect to main thread
        sys.stdout = stream_redirector
        sys.stderr = stream_redirector

        if self.use_pfam:
            used_model_name = P53_PFAM_MODEL_NAME
        else:
            used_model_name = P53_MODEL_NAME

        self.log_signal.emit(f"<b>Note:</b> Pfam use set to: {self.use_pfam}\n")

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
            data = p53_1_data_prep.p53_cleaning(data, pfam=self.use_pfam)

            # Add `Conservation` column to the dataset and drop `Domain` column
            if self.use_pfam:
                data = p53_1_data_prep.add_pfam_conservation(data)

            # Save the processed dataset (not encoded nor scaled) in CSV format
            data_save_path = save_processed_data(data, used_model_name) # Save the processed data Not Encoded nor Scaled
            if data_save_path:
                self.log_signal.emit(f"Processed non encoded dataset saved to:  \n{data_save_path}\n")
            else:
                self.log_signal.emit("Error saving processed non encoded dataset.")
            
            # Encode and scale the data
            self.log_signal.emit("Encoding the data...")
            data = p53_2_encoding.p53_encoding(data, pfam=self.use_pfam)
            
            self.log_signal.emit("Scaling the data...")
            data = p53_3_scaling.p53_scaling(data)

            # Save the processed dataset (encoded and scaled) in CSV format
            data_save_path = save_processed_data(data, used_model_name, is_encoded=True, ) # Save the processed data Encoded and Scaled but not Balanced
            if data_save_path:
                self.log_signal.emit(f"Processed Encoded dataset saved to:  \n{data_save_path}\n")
            else:
                self.log_signal.emit("Error saving processed Encoded dataset.")

            # Balance and Split the data
            self.log_signal.emit("Balancing the data...")
            X_resampled, y_resampled = p53_4_data_balancing.balance_split_data(data)

            self.log_signal.emit("Creating the training and test sets...")
            X_train, X_test, y_train, y_test \
                = p53_split.create_train_test_sets(X_data=X_resampled, y_labels=y_resampled)
            
            # Train the model
            self.log_signal.emit("Training the model...")
            model, history = p53_6_model.p53_train_model(X_train, y_train, X_test, y_test, pfam=self.use_pfam)

            # Evaluate the model
            self.log_signal.emit("Evaluating the model...")


            X_test = p53_6_model.input_dict_prepare(X_test, pfam=self.use_pfam)

            self.log_signal.emit("\nModel evaluation:")
            ev.simple_evaluate_model(model, X_test, y_test)

            self.log_signal.emit("\nSaving the model statistics...")
            ev.save_extended_stats(
                model=model, 
                X_test=X_test, 
                y_test=y_test,
                history=history,
                model_name=used_model_name
            )

            plot_acc = ev.plot_accuracy(history, model_name=used_model_name)            
            ev.save_plot_incremental(plot_acc, model_name=used_model_name)

            plot_loss = ev.plot_loss(history, model_name=used_model_name)
            ev.save_plot_incremental(plot_loss, model_name=used_model_name)

            plot_auc = ev.plot_curve_auc(model, X_test, y_test, model_name=used_model_name)
            ev.save_plot_incremental(plot_auc, model_name=used_model_name)



            self.log_signal.emit("\nCross-validation evaluation:")
            ev.n_times_k_fold_eval(model, X_resampled, y_resampled, n_splits=10, n_repeats=2, pfam=self.use_pfam)

            self.log_signal.emit("Training complete!\n ------------------- \n")

            # Train the model to save it
            self.log_signal.emit("Training the model with all the data to save it...")

            model, history = p53_6_model.retrain_model_to_save(model, X_resampled, y_resampled, pfam=self.use_pfam)

            # Save the model
            self.log_signal.emit("\nSaving the model...")

            p53_6_model.save_model(model, name=used_model_name)

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
    

    
    # -------------------------------------------- HRAS MODEL -------------------------------------------- #

    # TODO: This function is very similar to the P53 model training function.


    def __train_hras_model(self):
        """
            Train the hras model.
        """
        # Local imports to avoid circular imports
        import src.hras.hras_1_data as data_prep
        import src.hras.hras_2_encoding as encoding
        import src.hras.hras_3_scaling as scaling
        import src.hras.hras_4_data_balancing as balancing
        import src.hras.hras_5_data_split as split
        import src.hras.hras_6_model as hras_model

        # Redirect stdout and stderr to capture Keras output
        stream_redirector = StreamRedirector()
        stream_redirector.log_signal.connect(self.log_signal)  # Connect to main thread
        sys.stdout = stream_redirector
        sys.stderr = stream_redirector

        if self.use_pfam:
            used_model_name = HRAS_MODEL_NAME
        else:
            self.log_signal.emit(" -- Error: HRAS model requires pfam. -- ")
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            return
        

        self.log_signal.emit(f"<b>Note:</b> Pfam use set to: {self.use_pfam}\n")

        try:            
            # Load the P53 Model to use for Transfer Learning
            from_model = mu.load_model_by_real_name(P53_PFAM_MODEL_NAME)
            if from_model is None:
                self.log_signal.emit(" <b>Error:</b> No P53 Pfam Model found.")
                self.log_signal.emit(" <b>Important:</b> This is a Transfer Learning model. The P53 Pfam Model is required.")
                self.log_signal.emit("  Train and save the P53 Pfam Model first.")
                self.log_signal.emit("Terminated: P53 Pfam Model not found. -- ")
                return
            
            self.log_signal.emit("Loaded P53 Pfam Model for Transfer Learning.")

            self.log_signal.emit("Starting the training process...\n")

            data = load_data(protein = 'hras') # Loads saved data or downloads it if not present
            
            if data is None:
                self.log_signal.emit("Error loading data.")
                return
            
            self.log_signal.emit("Dataset loaded.")
            self.log_signal.emit(f"Data: {data.shape[0]} x {data.shape[1]}\n")

            # Preprocess the data
            self.log_signal.emit("Preprocessing the data...")
            
            self.log_signal.emit("Cleaning the data...")
            data = data_prep.hras_data_clean_extract(data, pfam=self.use_pfam, email=self.email)

            # Save the processed dataset (not encoded nor scaled) in CSV format
            data_save_path = save_processed_data(data, used_model_name) # Save the processed data Not Encoded nor Scaled
            if data_save_path:
                self.log_signal.emit(f"Processed non encoded dataset saved to:  \n{data_save_path}\n")
            else:
                self.log_signal.emit("Error saving processed non encoded dataset.")
            
            # Encode and scale the data
            self.log_signal.emit("Encoding the data...")
            data = encoding.hras_data_encoding(data, pfam=self.use_pfam)
            
            self.log_signal.emit("Scaling the data...")
            data = scaling.hras_scaling(data, pfam=self.use_pfam)

            # Save the processed dataset (encoded and scaled) in CSV format
            data_save_path = save_processed_data(data, used_model_name, is_encoded=True) # Save the processed data Encoded and Scaled but not Balanced
            if data_save_path:
                self.log_signal.emit(f"Processed Encoded dataset saved to:  \n{data_save_path}\n")
            else:
                self.log_signal.emit("Error saving processed Encoded dataset.")

            # Balance and Split the data
            self.log_signal.emit("Balancing the data...")
            X_resampled, y_resampled = balancing.hras_data_balancing(data)

            self.log_signal.emit("Creating the training and test sets...")
            X_train, X_test, y_train, y_test \
                = split.create_train_test_sets(X_data=X_resampled, y_labels=y_resampled)
            
            # Train the model
            self.log_signal.emit("Training the model...")
            model, history = \
                hras_model.hras_transfer_model_v2(
                    X_train=X_train, 
                    X_test=X_test,
                    y_train=y_train, 
                    y_test=y_test,
                    original_model=from_model,
                    pfam=self.use_pfam
                )
    
            # Evaluate the model
            self.log_signal.emit("Evaluating the model...")

            X_test = hras_model.input_dict_prepare(X_test, pfam=self.use_pfam)

            self.log_signal.emit("\nModel evaluation:")
            ev.simple_evaluate_model(model, X_test, y_test)

            self.log_signal.emit("\nSaving the model statistics...")
            ev.save_extended_stats(
                model=model, 
                X_test=X_test, 
                y_test=y_test,
                history=history,
                model_name=used_model_name
            )

            plot_acc = ev.plot_accuracy(history, model_name=used_model_name)            
            ev.save_plot_incremental(plot_acc, model_name=used_model_name)

            plot_loss = ev.plot_loss(history, model_name=used_model_name)
            ev.save_plot_incremental(plot_loss, model_name=used_model_name)

            plot_auc = ev.plot_curve_auc(model, X_test, y_test, model_name=used_model_name)
            ev.save_plot_incremental(plot_auc, model_name=used_model_name)



            self.log_signal.emit("\nCross-validation evaluation:")
            ev.n_times_k_fold_eval(model, X_resampled, y_resampled, n_splits=10, n_repeats=2, pfam=self.use_pfam)

            self.log_signal.emit("First Training complete!\n ------------------- \n")

            # Train the model to save it
            self.log_signal.emit("Training the model with all the data to save it...")

            model, history = hras_model.retrain_hras_model_to_save(
                X_resampled, y_resampled, model, use_P53_v2=True, pfam=self.use_pfam)
            

            # Save the model
            self.log_signal.emit("\nSaving the model...")
            mu.save_model(model, name=used_model_name)

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
    