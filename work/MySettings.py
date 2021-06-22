from PyQt5.Qt import QSettings


class MyQSettings(QSettings):
    def __init__(self):
        super(MyQSettings, self).__init__()
        self.settings = QSettings('./config.ini', QSettings.IniFormat)