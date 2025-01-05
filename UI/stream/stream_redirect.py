import sys
from PyQt5.QtCore import pyqtSignal, QObject

class StreamRedirector(QObject):
    """
    Redirects standard output and error streams to a PyQt signal.
    """
    log_signal = pyqtSignal(str)  # Signal to send log messages to the GUI

    def __init__(self):
        super().__init__()

    def write(self, message):
        # Emit the message as a signal
        self.log_signal.emit(message)

    def flush(self):
        # Required for compatibility with sys.stdout and sys.stderr
        pass
