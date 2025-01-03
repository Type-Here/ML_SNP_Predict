import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QLabel, QPushButton, QComboBox, QLineEdit, QTextEdit, QSpacerItem, QSizePolicy, QAction, QMessageBox
)
from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QIntValidator, QRegExpValidator, QIcon, QPixmap

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from ..src import MODELS_DIR, DATA_PATH, MODELS_STATS_DIR, PFAM_PATH
from src.fasta.fasta_seq import get_fasta_sequence_by_model_name
from src.models_usage import load_model_by_name, get_prediction
from src.config import MODELS_DIR, DATA_PATH, PFAM_PATH, MODELS_STATS_DIR

# Imposta dinamicamente il percorso QT_QPA_PLATFORM_PLUGIN_PATH
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(os.environ["CONDA_PREFIX"], "plugins", "platforms")
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Forza l'uso di xcb

from info import AUTHOR, VERSION
from info_dialog import InfoDialog

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

        self.setWindowIcon(QIcon("UI/assets/logo.png"))

        # Main Layout
        main_layout = QHBoxLayout()

        # Left Area (Predizione e Log)
        left_layout = QVBoxLayout()

        # Section: Add Email Address
        email_layout = QHBoxLayout()
        email_layout.setAlignment(Qt.AlignLeft)
        email_label = QLabel("Email*:")
        email_input = QLineEdit()
        email_label_explain = QLabel(" Richiesto da API EntreZ per il download della sequenza.")
        email_label_explain.setStyleSheet("color: gray; font-size: 10px;")
        email_input.setPlaceholderText("Inserisci la tua email")
        email_input.setMaximumWidth(200)
        email_input.setValidator(QRegExpValidator(
                    QRegExp(r"[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$"))
        )

        email_layout.addWidget(email_label)
        email_layout.addWidget(email_input)
        email_layout.addWidget(email_label_explain)
        email_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        left_layout.addLayout(email_layout)

        # Section: Load Model
        model_layout = QHBoxLayout()
        model_label = QLabel("Seleziona Modello:")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["P53 Model", "P53 Pfam", "HRAS Transfer"]) 
        load_model_button = QPushButton("Carica")
        load_model_button.setMaximumWidth(80)
        load_model_button.clicked.connect(self.load_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        model_layout.addWidget(load_model_button)

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
        self.predict_button.clicked.connect(self.predict)
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
        self.prediction_output.setReadOnly(True)
        self.prediction_output.setPlaceholderText("Dati di Predizione e Score")

        left_layout.addWidget(self.prediction_output)

        # Section: Log
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Log")

        left_layout.addWidget(self.log_output)

        main_layout.addLayout(left_layout, 2)  # 2/3 width

        # Right Area (Biopython 3D)
        right_layout = QVBoxLayout()
        self.biopython_view = QLabel("Biopython Proteina 3D")
        self.biopython_view.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.biopython_view)
        
        main_layout.addLayout(right_layout, 1)  # 1/3 width

        # Set the central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_model(self):

        self.active_model = self.model_dropdown.currentText()
        
        # Load the model
        self.model = self.__load_model()
        if self.model is None:
            self.log_output.append("Errore nel caricamento del modello.")
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
        
        # Enable the input fields
        self.position_input.setReadOnly(False)
        self.mut_dropdown.setDisabled(False)  
        self.predict_button.setDisabled(False)  
        self.prediction_output.clear()


    def predict(self):
        # Placeholder 
        position = self.position_input.text()
        ref = "A"  # TODO: Get the reference nucleotide from the dataset
        self.ref_nucleotide.setText(ref)
        mutation = self.mut_dropdown.currentText()
        self.prediction_output.append(f"Posizione: {position}, Ref: {ref}, Mut: {mutation}")
        self.log_output.append("Predizione effettuata con successo.")


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
        self.sequence = get_fasta_sequence_by_model_name(self.active_model) 
        return self.sequence
        

    def __get_prediction(self, position, ref, mutation):
        if not self.model:
            return None
        prediction = get_prediction(
            model_name=self.active_model, 
            model=self.model, 
            position=position, 
            ref=ref, 
            mut=mutation, 
            sequence=self.sequence
        )
        return prediction
        
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
