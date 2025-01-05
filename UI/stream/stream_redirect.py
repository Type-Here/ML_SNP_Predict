import sys

from PyQt5.QtCore import pyqtSignal, QObject
from ansi2html import Ansi2HTMLConverter

class StreamRedirector(QObject):
    """
    Redirects standard output and error streams to a PyQt signal.
    """
    log_signal = pyqtSignal(str)  # Signal to send log messages to the GUI

    def __init__(self):
        self.converter = Ansi2HTMLConverter()
        super().__init__()

    def write(self, message):
        # Emit the message as a signal
        
        # Convert the message to HTML
        html_text = self.converter.convert(message, full=False)
        
        self.log_signal.emit(html_text)

    def flush(self):
        # Required for compatibility with sys.stdout and sys.stderr
        pass
