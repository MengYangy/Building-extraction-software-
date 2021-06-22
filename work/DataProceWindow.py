from PyQt5.Qt import QDialog, QErrorMessage, QLabel, Qt, QApplication, QAction, QIcon, pyqtSignal, \
    QTreeWidget,QGridLayout, QFileDialog, QSettings,QSpinBox, QMessageBox, \
    QGraphicsScene, QPushButton, QToolButton, QHBoxLayout, QWidget, QThread, QFormLayout, QLineEdit
from PyQt5 import QtCore, QtGui
import sys, time
from Algo.DataPro import data_pro
from TrainModelWindow import MyQSpinBox
from MySettings import MyQSettings


class DataProcess_Thread(QThread):
    def __init__(self):
        super(DataProcess_Thread, self).__init__()

    def run(self):
        try:
            # myPrint(1)
            data_pro(self.img_path, self.lab_path, self.save_img_path, self.save_lab_path,
                     self.cut_num, self.cut_width, self.cut_height)
        except Exception as e:
            print('错误原因是：' + str(e))

    def get_parameters(self, img_path, lab_path, save_img_path, save_lab_path, cut_num, cut_width, cut_height):
        self.img_path = img_path
        self.lab_path = lab_path
        self.save_img_path = save_img_path
        self.save_lab_path = save_lab_path
        self.cut_num = cut_num
        self.cut_width = cut_width
        self.cut_height = cut_height


class Data_pro_win(QWidget):
    def __init__(self, parent=None):
        super(Data_pro_win, self).__init__(parent)
        self.setWindowTitle('数据增强')
        # self.setStyleSheet('background-color:rgb(255,255,255)')
        self.setFixedSize(400, 500)
        self.setContentsMargins(20, 20, 20, 20)
        self.setWindowFlags(Qt.Window)
        self.unsetCursor()

        self.isStarting = False
        self.my_Setting = MyQSettings()


        self.build()
        self.data_thread = DataProcess_Thread()
        # print(dir(self.data_thread))


    def build(self):
        gridlayout = QGridLayout()
        hbox = QHBoxLayout()
        btn_hbox = QHBoxLayout()

        img_path_lab = QLabel('训练图像', self)
        self.img_path_line = QLineEdit()
        self.img_path_line.setReadOnly(True)
        img_path_btn = QToolButton()
        img_path_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        img_path_btn.setToolTip('选择训练样本')

        lab_path_lab = QLabel('标签图像', self)
        self.lab_path_line = QLineEdit()
        self.lab_path_line.setReadOnly(True)
        lab_path_btn = QToolButton()
        lab_path_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        lab_path_btn.setToolTip('选择训练标签')

        cut_num_lab = QLabel('裁剪数量', self)
        self.cut_num_spin = MyQSpinBox(self)
        self.cut_num_spin.setValue(int(self.my_Setting.settings.value('cut_num', type=int)))
        # print(self.my_Setting.settings.value('cut_num', type=int))
        # print(type(self.my_Setting.settings.value('cut_num', type=int)))

        cut_size_lab = QLabel('裁剪尺寸')
        self.cut_width_spin = MyQSpinBox()
        self.cut_width_spin.setPrefix('宽：')
        self.cut_width_spin.setValue(int(self.my_Setting.settings.value('cut_w')))

        self.cut_height_spin = MyQSpinBox()
        self.cut_height_spin.setPrefix('高：')
        self.cut_height_spin.setValue(int(self.my_Setting.settings.value('cut_h')))

        save_img_path_lab = QLabel('保存图像', self)
        self.save_img_path_line = QLineEdit()
        self.save_img_path_line.setReadOnly(True)
        save_img_path_btn = QToolButton()
        save_img_path_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        save_img_path_btn.setToolTip('保存训练样本')

        save_lab_path_lab = QLabel('保存标签', self)
        self.save_lab_path_line = QLineEdit()
        self.save_lab_path_line.setReadOnly(True)
        save_lab_path_btn = QToolButton()
        save_lab_path_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        save_lab_path_btn.setToolTip('保存训练标签')

        start_btn = QPushButton(QIcon('../Res/VectorEditor_StartEdit.png'), '开始', self)
        stop_btn = QPushButton(QIcon('../Res/VectorEditor_StopEdit.png'), '取消', self)

        gridlayout.addWidget(img_path_lab, 1, 0)
        gridlayout.addWidget(self.img_path_line, 1, 1, 1, 2)
        gridlayout.addWidget(img_path_btn, 1, 3)

        gridlayout.addWidget(lab_path_lab, 2, 0)
        gridlayout.addWidget(self.lab_path_line, 2, 1, 1, 2)
        gridlayout.addWidget(lab_path_btn, 2, 3)

        gridlayout.addWidget(cut_num_lab, 3, 0)
        gridlayout.addWidget(self.cut_num_spin, 3, 1)

        hbox.addWidget(self.cut_width_spin)
        hbox.addWidget(self.cut_height_spin)
        gridlayout.addWidget(cut_size_lab, 4, 0)
        gridlayout.addLayout(hbox, 4, 1)

        gridlayout.addWidget(save_img_path_lab, 5, 0)
        gridlayout.addWidget(self.save_img_path_line, 5, 1, 1, 2)
        gridlayout.addWidget(save_img_path_btn, 5, 3)

        gridlayout.addWidget(save_lab_path_lab, 6, 0)
        gridlayout.addWidget(self.save_lab_path_line, 6, 1, 1, 2)
        gridlayout.addWidget(save_lab_path_btn, 6, 3)

        btn_hbox.addStretch(1)
        btn_hbox.addWidget(start_btn)
        btn_hbox.addWidget(stop_btn)
        gridlayout.addLayout(btn_hbox, 7, 1)

        self.setLayout(gridlayout)

        """         各个信号         """
        img_path_btn.clicked.connect(self.select_img_path_slot_func)
        lab_path_btn.clicked.connect(self.select_lab_path_slot_func)
        save_img_path_btn.clicked.connect(self.select_save_img_path_slot_func)
        save_lab_path_btn.clicked.connect(self.select_save_lab_path_slot_func)
        start_btn.clicked.connect(self.start_slot_func)
        stop_btn.clicked.connect(self.stop_slot_func)

        # print(self.cut_height_spin.text().split('：')[-1])

    def select_img_path_slot_func(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getExistingDirectory(self, '选择训练图像样本目录')
        self.img_path_line.setText(file_path)
        # print(file_path)

    def select_lab_path_slot_func(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getExistingDirectory(self, '选择训练标签样本目录')
        self.lab_path_line.setText(file_path)

    def select_save_img_path_slot_func(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getExistingDirectory(self, '选择训练标签样本目录')
        self.save_img_path_line.setText(file_path)

    def select_save_lab_path_slot_func(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getExistingDirectory(self, '选择训练标签样本目录')
        self.save_lab_path_line.setText(file_path)

    def start_slot_func(self):
        error_text = ''
        if self.img_path_line.text() == '':
            error_text += '训练图像样本目录为空\n'

        if self.lab_path_line.text() == '':
            error_text += '训练标签样本目录为空\n'

        if self.save_img_path_line.text() == '':
            error_text += '保存图像目录为空\n'

        if self.save_lab_path_line.text() == '':
            error_text += '保存标签目录为空\n'

        if self.cut_num_spin.value() == 0:
            error_text += '裁剪数量为0\n'

        if self.cut_width_spin.value() == 0 or self.cut_height_spin.value() == 0:
            error_text += '裁剪尺寸出现错误\n'

        if error_text != '':
            dialog = QErrorMessage()
            dialog.showMessage(error_text)
            dialog.exec()
            return None

        if self.isStarting == False and self.data_thread.isRunning() == False:
            self.isStarting = True
            self.data_thread.get_parameters(self.img_path_line.text(), self.lab_path_line.text(),
                                            self.save_img_path_line.text(), self.save_lab_path_line.text(),
                                            self.cut_num_spin.value(), self.cut_width_spin.value(),
                                            self.cut_height_spin.value())

            self.data_thread.start()
            self.data_thread.finished.connect(self.my_thread_finished)
            print('start data processing')

    def my_thread_finished(self):
        self.isStarting = False
        print('finished data process')

    def stop_slot_func(self):
        self.isStarting = False
        if self.data_thread.isRunning():
            try:
                self.data_thread.quit()
                print('stop data process')
                self.data_thread.terminate()
                self.data_thread.wait()
            except Exception as e:
                print('出现错误 --> 错误原因是：' + str(e))

    def save_settings(self):
        self.my_Setting.settings.setValue('cut_num', self.cut_num_spin.value())
        self.my_Setting.settings.setValue('cut_w', self.cut_width_spin.value())
        self.my_Setting.settings.setValue('cut_h', self.cut_height_spin.value())
        # self.my_Setting.settings.setValue('resolution', '0.01')
        self.my_Setting.settings.sync()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if self.data_thread.isRunning():
            a0.ignore()
            mb = QMessageBox(QMessageBox.Information, '提示', '请点击取消按钮，关闭当前进程后退出', QMessageBox.Ok)
            mb.exec()
        else:
            self.save_settings()
            if self.data_thread.isFinished():
                del self.data_thread


if __name__ == '__main__':
    app = QApplication(sys.argv)
    data_pro_w = Data_pro_win()
    data_pro_w.show()
    sys.exit(app.exec())
