import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QLabel, QPushButton, QComboBox, QLineEdit, QTextEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from ..src import MODELS_DIR, DATA_PATH, MODELS_STATS_DIR, PFAM_PATH
from src.fasta.fasta_seq import get_fasta_sequence_by_model_name
from src.load_models import load_model_by_name
from src.config import MODELS_DIR, DATA_PATH, PFAM_PATH, MODELS_STATS_DIR

# Imposta dinamicamente il percorso QT_QPA_PLATFORM_PLUGIN_PATH
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(os.environ["CONDA_PREFIX"], "plugins", "platforms")
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Forza l'uso di xcb


# Create Models, Pfam and Data Directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MODELS_STATS_DIR, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(PFAM_PATH, exist_ok=True)

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Model Predictor")
        self.setGeometry(100, 100, 1200, 800)

        # Main Layout
        main_layout = QHBoxLayout()

        # Left Area (Predizione e Log)
        left_layout = QVBoxLayout()

        # Section: Load Model
        model_layout = QHBoxLayout()
        model_label = QLabel("Seleziona Modello:")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["P53 Model", "P53 Pfam", "HRAS Transfer"]) 
        load_model_button = QPushButton("Carica")
        load_model_button.clicked.connect(self.load_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        model_layout.addWidget(load_model_button)
        left_layout.addLayout(model_layout)

        # Section: Prediction
        predict_layout = QHBoxLayout()
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

        predict_layout.addWidget(QLabel("Posizione"))
        predict_layout.addWidget(self.position_input)
        predict_layout.addWidget(QLabel("Ref"))
        predict_layout.addWidget(self.ref_nucleotide)
        predict_layout.addWidget(self.mut_dropdown)
        predict_layout.addWidget(self.predict_button)
        left_layout.addLayout(predict_layout)

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

        # Righe Area (Biopython 3D)
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
        # Load the model
        self.model = self.__load_model()
        if self.model is None:
            self.log_output.append("Errore nel caricamento del modello.")
            return
        
        self.log_output.clear()
        self.log_output.append("Modello caricato: " + self.model_dropdown.currentText())
        
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


    def __PositionValidator(self, min=1, max=9999):
        return QIntValidator(min, max)
    

    def __load_model(self):
        selected = self.model_dropdown.currentText()
        self.model = load_model_by_name(selected)
        return self.model
    

    def __load_sequence(self):
        selected = self.model_dropdown.currentText()
        self.sequence = get_fasta_sequence_by_model_name(selected) 
        return self.sequence
        
        
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
