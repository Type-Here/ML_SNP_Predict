import PyQt5
print(PyQt5.__file__)

from PyQt5.QtCore import QLibraryInfo
print(QLibraryInfo.location(QLibraryInfo.PluginsPath))
