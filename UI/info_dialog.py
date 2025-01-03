from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import os

from info import AUTHOR, VERSION, LOGO_PATH

class InfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Info")
        self.setFixedSize(400, 300) # Window size

        # Main layout
        main_layout = QVBoxLayout()

        # Logo
        absolute_path = os.path.abspath(LOGO_PATH)
        logo_label = QLabel(self)
        pixmap = QPixmap(absolute_path).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(logo_label)

        # Info text
        text_label = QLabel(self)
        text_label.setText(
            "<h2>GeneInsight</h2>"
            "<b>A DNA Mutation Prediction App</b><br><br>"
            "<p>An app to predict the effect of DNA mutations on protein sequences.</p>"
            f"<br><b>Version:</b> {VERSION}<br>"
            f"<b>Developed by:</b> {AUTHOR}<br>"
        )
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setWordWrap(True)  # Wrap text
        main_layout.addWidget(text_label)

        # Closing button
        button_layout = QHBoxLayout()
        close_button = QPushButton("Ok")
        close_button.clicked.connect(self.accept)  # Close the dialog
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

# Usage
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    dialog = InfoDialog()
    dialog.exec_()
