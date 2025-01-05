import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QLabel, QPushButton, QComboBox, QLineEdit, QTextEdit, QSpacerItem, QSizePolicy, QAction, 
    QMessageBox, QStackedWidget, QToolButton, QSplitter, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QIntValidator, QRegExpValidator, QIcon, QPixmap

import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.p53_model_train import TrainingThread #TODO Generalize the TrainingThread to be used for all models

#from ..src import MODELS_DIR, DATA_PATH, MODELS_STATS_DIR, PFAM_PATH
from src.fasta.fasta_seq import get_fasta_sequence_by_model_name
from src.models_usage import load_model_by_name, get_prediction
from src.config import MODELS_DIR, DATA_PATH, PFAM_PATH, MODELS_STATS_DIR

# Set dinamically the path for QT_QPA_PLATFORM_PLUGIN_PATH
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(os.environ["CONDA_PREFIX"], "plugins", "platforms")
if sys.platform == "linux":
    os.environ["QT_QPA_PLATFORM"] = "xcb"  # Fix for Linux: Could not find a Qt installation of ''; Force to use xcb

from info import AUTHOR, VERSION, LOGO_PATH, STYLE_PATH
from info_dialog import InfoDialog
from stats_view import display_plots_in_layout, display_stats_in_textedit, get_model_name_from_common_name

# Create Models, Pfam and Data Directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MODELS_STATS_DIR, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(PFAM_PATH, exist_ok=True)

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeneInsight")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        self.setWindowIcon(QIcon(LOGO_PATH))

        # Main Layout
        main_layout = QHBoxLayout()
        main_layout.setObjectName("main_layout")

        # Splitter for the left and right areas
        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("splitter")

        # ------ Main Left Area (Prediction and Log) ------ #
        left_layout = QVBoxLayout()

        self.__set_left_layout(left_layout)

        #main_layout.addLayout(left_layout, 2)  # 2/3 width
        
        # Container Widget to hold the left layout
        left_container = QWidget()
        left_container.setObjectName("left_container")
        left_container.setLayout(left_layout)

        # Add the left container to the splitter
        splitter.addWidget(left_container)


        # ------ Right Area (Biopython 3D) ------ #
        self.right_layout = QVBoxLayout()
        
        # This function will be used to set the right layout content
        self.__set_right_layout()
       
        #main_layout.addLayout(self.right_layout, 1)  # 1/3 width

        # Container Widget to hold the right layout
        right_container = QWidget()
        right_container.setObjectName("right_container")
        right_container.setLayout(self.right_layout)
        
        # Add the right container to the splitter
        splitter.addWidget(right_container)
        splitter.setSizes([600, 200])

        main_layout.addWidget(splitter)

        # Set the central widget
        container = QWidget()
        container.setObjectName("main_container")
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Create the menu
        self.__create_menu()

        # Load the stylesheet
        self.__load_stylesheet()


    # ---- GUI Functions ---- #

    def __load_stylesheet(self):
        try:
            with open(STYLE_PATH, "r") as file:
                self.setStyleSheet(file.read())
        except FileNotFoundError:
            QMessageBox.warning(self, "Errore", "File di stile non trovato.")
        except Exception as e:
            QMessageBox.warning(self, "Errore", f"Errore nel caricamento dello stile: {e}")


    def __create_menu(self):
        """
            Create the menu bar.
        """
        # Menu Bar
        menubar = self.menuBar()

        # Menu File
        file_menu = menubar.addMenu("File")
        exit_action = QAction("Esci", self)
        exit_action.triggered.connect(self.close)  #Close the application
        file_menu.addAction(exit_action)

        # Menu About
        about_menu = menubar.addMenu("About")
        info_action = QAction("Info", self)
        info_action.triggered.connect(self.__show_info_dialog)  # Show dialog Info
        about_menu.addAction(info_action)


    def __show_info_dialog(self):
        dialog = InfoDialog(self)
        dialog.exec_()


    # ---------- Left Layout ---------- #

    def __set_left_layout(self, left_layout):
        """
            Set the left layout content.
        """
        
        # Section: Add Email Address
        email_layout = QHBoxLayout()
        email_layout.setAlignment(Qt.AlignLeft)
        email_label = QLabel("Email*:")
        self.email_input = QLineEdit()
        email_label_explain = QLabel(" Richiesto da API EntreZ per il download della sequenza. Non necessaria registrazione.")
        email_label_explain.setStyleSheet("color: gray; font-size: 10px;")
        self.email_input.setPlaceholderText("Inserisci la tua email")
        self.email_input.setMaximumWidth(200)
        self.email_input.setValidator(QRegExpValidator(
                    QRegExp(r"[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$"))
        )

        email_layout.addWidget(email_label)
        email_layout.addWidget(self.email_input)
        email_layout.addWidget(email_label_explain)
        email_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        left_layout.addLayout(email_layout)

        # Section: Load Model
        model_layout = QHBoxLayout()
        model_label = QLabel("Seleziona Modello:")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["P53 Model", "P53 Pfam", "HRAS Transfer"]) 
        self.load_model_button = QPushButton("Carica")
        self.load_model_button.setMaximumWidth(80)
        self.load_model_button.clicked.connect(self.load_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        model_layout.addWidget(self.load_model_button)

        left_layout.addLayout(model_layout)

        # Section: Prediction
        predict_layout = QHBoxLayout()
        predict_layout.setAlignment(Qt.AlignLeft)
        self.position_input = QLineEdit()
        self.position_input.setPlaceholderText("Posizione")
        self.position_input.setMaximumWidth(150)
        self.position_input.setValidator(self.__PositionValidator())
        self.position_input.setReadOnly(True)

        self.ref_nucleotide = QLineEdit()
        self.ref_nucleotide.setReadOnly(True)
        self.ref_nucleotide.setPlaceholderText("Ref")
        self.ref_nucleotide.setMaximumWidth(50)
        self.ref_nucleotide.setReadOnly(True)

        self.mut_dropdown = QComboBox()
        self.mut_dropdown.setMaximumWidth(40)
        self.mut_dropdown.addItems(["A", "T", "C", "G"])
        self.mut_dropdown.setPlaceholderText("Mut")
        self.mut_dropdown.setDisabled(True)

        self.predict_button = QPushButton("Predici")
        self.predict_button.clicked.connect(self.__predict)
        self.predict_button.setMaximumWidth(80)
        self.predict_button.setDisabled(True)
        
        pos_label = QLabel("Posizione")
        pos_label.setStyleSheet("width: min-content; text-align: right;")
        pos_label.setMaximumWidth(60)
        ref_label = QLabel("Ref")
        ref_label.setMaximumWidth(30)
        mut_label = QLabel("Mut")
        mut_label.setMaximumWidth(30)


        predict_layout.addWidget(pos_label)
        predict_layout.addWidget(self.position_input)
        predict_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        predict_layout.addWidget(ref_label)
        predict_layout.addWidget(self.ref_nucleotide)
        predict_layout.addWidget(mut_label)
        predict_layout.addWidget(self.mut_dropdown)
        predict_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        predict_layout.addWidget(self.predict_button)
        predict_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        left_layout.addLayout(predict_layout)

         # Listener for position input to update the ref nucleotide on change
        self.position_input.textChanged.connect(self.__update_ref_nucleotide)

        # Section: Prediction Output
        self.prediction_output = QTextEdit()
        self.prediction_output.setAcceptRichText(True)
        self.prediction_output.setReadOnly(True)
        self.prediction_output.setPlaceholderText("Dati di Predizione e Score")

        left_layout.addWidget(self.prediction_output)

        # Section: Log
        self.log_output = QTextEdit()
        self.log_output.setAcceptRichText(True) 
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Log")
        self.log_output.setMaximumHeight(200)

        left_layout.addWidget(self.log_output)

    
    # ---------- Right Layout ---------- #

    def __set_right_layout(self):
        """
            Set the right layout content.
        """
        self.right_layout.setContentsMargins(0, 0, 0, 0) 

         # Spacer per allineare con il prediction output
        spacer = QSpacerItem(80, 100, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.right_layout.addSpacerItem(spacer)
       
        # Accordion: Section: Biopython 3D #
        
        # Accordion Byopython Button
        biopython_button = QToolButton()
        biopython_button.setText("Biopython 3D")
        biopython_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        biopython_button.setObjectName("biopython_3d_button")
        biopython_button.setCheckable(True)
        biopython_button.setChecked(False)
        self.right_layout.addWidget(biopython_button)

        # Biopython Content
        biopython_content = QFrame()
        biopython_content.setObjectName("biopython_frame")
        biopython_content.setFrameShape(QFrame.Box)

        biopython_layout = QVBoxLayout()
        biopython_layout.setObjectName("biopython_layout")
        biopython_layout.setContentsMargins(1, 1, 1, 1) 
        
        biopython_label = QLabel("Contenuto Biopython Proteina 3D")
        biopython_label.setAlignment(Qt.AlignCenter)

        # Scroll area for the QFrame
        bio_scroll_area = QScrollArea()
        bio_scroll_area.setWidget(biopython_content)  
        bio_scroll_area.setWidgetResizable(True)
        bio_scroll_area.setMinimumHeight(300)
        bio_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        bio_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        
        biopython_layout.addWidget(biopython_label)
        biopython_content.setLayout(biopython_layout)
        
        # Connect the button to the function to show/hide the content
        biopython_button.clicked.connect(lambda: self.__toggle_section(bio_scroll_area, biopython_button))
        bio_scroll_area.setVisible(False) # Hide the content by default

        self.right_layout.addWidget(bio_scroll_area)

        # Accordion: Setion: Model Stats #

        # Model Stats Button
        stats_button = QToolButton()
        stats_button.setText("Grafici Statistica Modello")
        stats_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        stats_button.setObjectName("model_stats_button")
        stats_button.setCheckable(True)
        stats_button.setChecked(False)
        self.right_layout.addWidget(stats_button)

        # Model Stats Content
        stats_content = QFrame()
        stats_content.setObjectName("stats_frame")
        stats_content.setFrameShape(QFrame.Box)

        stats_layout = QVBoxLayout()
        stats_layout.setObjectName("stats_layout")
        stats_layout.setContentsMargins(1, 1, 1, 1) 
        
        stats_label = QLabel("Contenuto Grafici Statistica Modello")
        stats_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(stats_label)

        # Scroll area for the QFrame
        stat_scroll_area = QScrollArea()
        stat_scroll_area.setWidget(stats_content)  
        stat_scroll_area.setWidgetResizable(True)
        stat_scroll_area.setMinimumHeight(300)
        stat_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        stat_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Connect the button to the function to show/hide the content
        stats_button.clicked.connect(lambda: self.__toggle_section(stat_scroll_area, stats_button))
        stat_scroll_area.setVisible(False)

        stats_content.setLayout(stats_layout)
        self.right_layout.addWidget(stat_scroll_area)

        # Accordion: Spacer #
        self.right_layout.addSpacerItem(QSpacerItem(10, 100, QSizePolicy.Minimum, QSizePolicy.Expanding))


    def __toggle_section(self, section, button):
        """Mostra o nasconde una sezione accordion."""
        section.setVisible(button.isChecked())
    

    def __disable_all_inputs(self):
        """
        Disable all input fields and buttons.
        """
        self.load_model_button.setDisabled(True)
        self.position_input.setReadOnly(True)
        self.mut_dropdown.setDisabled(True)
        self.predict_button.setDisabled(True)

    def __clear_layout(self, layout):
        """
            Clear the layout by removing all widgets and layouts.
        """
        while layout.count():
            item = layout.takeAt(0)  # Remove the item at index 0
            widget = item.widget()  # Check if the item has a widget
            if widget is not None:
                widget.deleteLater()  # Delete the widget
            else:
                layout_item = item.layout()  # Check if the item has a layout
                if layout_item is not None:
                    self.__clear_layout(layout_item)  # Delete the layout recursively
            


    # --- Train Dialog Functions --- #

    def __handle_missing_model(self, text_edit):
        """
        Handle the case where a saved model is not found.
        Ask the user if they want to train the model and manage the training process.

        Parameters:
            text_edit (QTextEdit): The text field for displaying log messages.
            training_thread (QThread): The thread responsible for training the model.
        """
        # Create a message box to ask the user
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("Modello Non Trovato")
        msg_box.setText("Nessun modello salvato per il tipo scelto. Vuoi addestrarne uno?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)

        # Show the dialog and get the user's response
        response = msg_box.exec_()

        if response == QMessageBox.Yes:

            try:
                # Create and start the training thread
                self.training_thread = TrainingThread()
            except Exception as e:
                text_edit.append(f"Error creating the training thread: {e}")
                return

            # Disable the load button during training
            self.__disable_all_inputs()

            # Clear the text field
            text_edit.clear()
            
            # Connect the training thread signals to update the text field
            self.training_thread.log_signal.connect(text_edit.append)
            
            # Re-enable the button when the thread finishes
            self.training_thread.finished.connect(lambda: self.load_model_button.setEnabled(True))
            
            # Start the training thread
            self.training_thread.start()
        else:
            # Close the dialog if the user chooses "No"
            msg_box.close()


    # ---- Operative Functions ---- #

    def load_model(self):
        # Get User Input
        self.prediction_output.clear()
        self.active_model = self.model_dropdown.currentText()
        self.email = self.email_input.text()

        # Check if email is valid
        if not self.email_input.hasAcceptableInput():
            QMessageBox.warning(self, "Email non valida", "Per favore, inserisci un'email valida.")
            return
        
        # Load the model
        self.model = self.__load_model()
        if self.model is None:
            self.log_output.append("Modello non trovato.")
            self.__handle_missing_model(self.prediction_output)
            return
        
        self.log_output.clear()
        self.log_output.append("Modello caricato: " + self.active_model)
        
        # Load DNA sequence
        self.sequence = self.__load_sequence()
        if self.sequence is None:
            self.log_output.append("Errore nel caricamento della sequenza.")
            return
        
        self.log_output.append("Sequenza caricata.")
        self.log_output.append("Inserisci la posizione, la mutazione e premi 'Predici'.")

        self.position_input.setValidator(
            self.__PositionValidator(min=1, max=len(self.sequence)))
        
        self.position_input.setPlaceholderText(f"Posizione (1-{len(self.sequence)})")
        
        # Load Model Stats
        self.__load_stats()
        
        # Enable the input fields
        self.position_input.setReadOnly(False)
        self.mut_dropdown.setDisabled(False)  
        self.predict_button.setDisabled(False)  


    def __update_ref_nucleotide(self):
        position = self.position_input.text()
        if position.isdigit():
            pos = int(position) - 1  # 0-based index
            if 0 <= pos < len(self.sequence):
                self.ref_nucleotide.setText(self.sequence[pos])
            else:
                self.ref_nucleotide.setText("")  # Out of range
        else:
            self.ref_nucleotide.setText("")  # Not a number


    def __PositionValidator(self, min=1, max=9999):
        return QIntValidator(min, max)
    

    def __load_model(self):
        if not self.active_model:
            return None
        self.model = load_model_by_name(self.active_model)
        return self.model
    

    def __load_sequence(self):
        if not self.active_model:
            return None
        self.sequence = get_fasta_sequence_by_model_name(self.active_model, self.email) 
        return self.sequence
    

    def __load_stats(self):
        if not self.active_model:
            return None
        # Load the stats

        model = get_model_name_from_common_name(self.active_model)
        stats_file = os.path.join(MODELS_STATS_DIR, f"{model}_stats.json")
        # Display the stats in the text edit
        display_stats_in_textedit(
            stats_file=stats_file,
            text_edit=self.prediction_output
        )
        
        self.log_output.append("Statistiche caricate.")
        self.prediction_output.append("-- Fine Statistiche --")

        stat_layout=self.findChild(QFrame, "stats_frame").layout()
        self.__clear_layout(stat_layout)

        # Display the plots in the layout
        display_plots_in_layout(
            stats_file=stats_file,
            layout=stat_layout,
            text_edit=self.prediction_output
        )

        # Close Biopython 3D section and open Model Stats section
        bio_but = self.findChild(QToolButton, "biopython_3d_button")
        if bio_but.isChecked():
            bio_but.click()
        sbut = self.findChild(QToolButton, "model_stats_button")
        if not sbut.isChecked():
            sbut.click()


    # ---- Prediction Functions ---- #

    def __predict(self):
        """
            Predict the mutation.
            Input Data:
                - position: The position of the mutation.
                - ref: The reference nucleotide.
                - mutation: The mutated nucleotide.
            Output:
                - The prediction of the model if successful. 

        """
        self.prediction_output.append(" ----------------- \n")
        
        # Get Input Data
        position = self.position_input.text()
        ref = self.ref_nucleotide.text()
        mutation = self.mut_dropdown.currentText()

        if not position or not ref or not mutation:
            self.prediction_output.append("<b>Errore: Inserisci la posizione e la mutazione.</b>")
            return


        self.prediction_output.append(f"Predizione per:")
        self.prediction_output.append(f"- Posizione: {position}, Ref: {ref}, Mut: {mutation}")
        
        # Get Prediction
        results = self.__get_prediction(position, ref, mutation)
        if results is None:
            self.prediction_output.append("<b>Errore nella predizione.</b>")
            return
        probabilities, label = results
        
        self.prediction_output.append(f"<b>Predizione Modello:</b> {position}")
        self.prediction_output.append(f"- Label mutazione predetta: <b>{label}</b>")
        self.prediction_output.append(f"- Probabilità: {probabilities}")
        

    def __get_prediction(self, position, ref, mutation):
        if not self.model:
            return None
        return get_prediction(
            model_name=self.active_model, 
            model=self.model, 
            position=position, 
            ref=ref, 
            mut=mutation, 
            sequence=self.sequence
        )
        

    # ---- End of Prediction Functions ---- #
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
