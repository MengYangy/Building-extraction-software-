from PyQt5.Qt import QSpinBox
from PyQt5 import QtGui


class MyQSpinBox(QSpinBox):
    def __init__(self, *args, **kwargs):
        super(MyQSpinBox, self).__init__(*args, **kwargs)
        self.setMaximum(9999)
        self.setFixedWidth(80)

    def focusInEvent(self, e: QtGui.QFocusEvent) -> None:
        self.grabKeyboard()
        QSpinBox.focusInEvent(self, e)

    def focusOutEvent(self, e: QtGui.QFocusEvent) -> None:
        self.releaseKeyboard()
        self.clearFocus()
        QSpinBox.focusOutEvent(self, e)