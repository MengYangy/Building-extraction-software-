from PyQt5.Qt import QLineEdit


class MyQLineEdit(QLineEdit):
    def __init__(self):
        super(MyQLineEdit, self).__init__()

    def textChanged(self, a0: str) -> None:
        print('文本发生变化')
        self.setToolTip(self.text())
        QLineEdit.textChanged(self, a0)



