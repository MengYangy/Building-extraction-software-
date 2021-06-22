from PyQt5.Qt import QDialog, QErrorMessage, QLabel, Qt, QApplication, QAction, QIcon, pyqtSignal, \
    QTreeWidget,QGridLayout, QFileDialog, QVBoxLayout,QSpinBox, QComboBox, \
    QGraphicsScene, QPushButton, QToolButton, QHBoxLayout, QWidget, QThread, QMessageBox, QLineEdit
from PyQt5 import QtCore, QtGui
import sys, time, os
from Algo import AgainTrain
from MySpinBox import MyQSpinBox
from MySettings import MyQSettings


class AgainTrain_Thread(QThread):
    def __init__(self):
        super(AgainTrain_Thread, self).__init__()

    def run(self) -> None:
        try:
            again_train = AgainTrain.train_model(self.img_path, self.lab_path, self.save_model_path,
                self.add_epoch_num, self.batch_size, self.original_epoch_num,self.current_model)
        except Exception as e:
            print('出现错误，原因是：' + str(e))

    def get_parameters(self, img_path, lab_path, save_model_path, original_epoch_num, batch_size,
                       cut_width, cut_height, add_epoch_num, current_model):
        self.img_path = img_path
        self.lab_path = lab_path
        self.save_model_path = save_model_path
        self.original_epoch_num = original_epoch_num
        self.batch_size = batch_size
        self.cut_width = cut_width
        self.cut_height = cut_height
        self.add_epoch_num = add_epoch_num
        self.current_model = current_model


class Again_Train_Model_Win(QWidget):
    def __init__(self):
        super(Again_Train_Model_Win, self).__init__()
        self.setWindowTitle('模型再训练界面')
        self.setFixedSize(400, 500)
        self.setContentsMargins(20, 20, 20, 20)
        self.setWindowFlags(Qt.Window)

        self.isStarting = False
        self.again_train_thread = AgainTrain_Thread()
        self.again_train_thread.finished.connect(self.my_thread_finished)

        self.my_Setting = MyQSettings()
        self.build()

    def build(self):
        gridlayout = QGridLayout()
        gridlayout.setContentsMargins(0, 0, 0, 0)
        hbox = QHBoxLayout()
        epoch_hbox = QHBoxLayout()
        btn_hbox = QHBoxLayout()

        img_path_lab = QLabel('训练图像', self)
        self.img_path_line = QLineEdit()
        self.img_path_line.setReadOnly(True)
        img_path_btn = QToolButton()
        img_path_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        img_path_btn.setToolTip('选择训练图像')
        gridlayout.addWidget(img_path_lab, 1, 0)
        gridlayout.addWidget(self.img_path_line, 1, 1, 1, 2)
        gridlayout.addWidget(img_path_btn, 1, 3)

        lab_path_lab = QLabel('训练标签', self)
        self.lab_path_line = QLineEdit()
        self.lab_path_line.setReadOnly(True)
        lab_path_btn = QToolButton()
        lab_path_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        lab_path_btn.setToolTip('选择训练标签')
        gridlayout.addWidget(lab_path_lab, 2, 0)
        gridlayout.addWidget(self.lab_path_line, 2, 1, 1, 2)
        gridlayout.addWidget(lab_path_btn, 2, 3)

        select_model_lab = QLabel('选择模型', self)
        self.select_model_line = QLineEdit()
        self.select_model_line.setReadOnly(True)
        select_model_btn = QToolButton()
        select_model_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        select_model_btn.setToolTip('选择模型')
        gridlayout.addWidget(select_model_lab, 3, 0)
        gridlayout.addWidget(self.select_model_line, 3, 1, 1, 2)
        gridlayout.addWidget(select_model_btn, 3, 3)

        cut_size_lab = QLabel('输入尺寸')
        self.cut_width_spin = MyQSpinBox()
        self.cut_width_spin.setPrefix('宽：')
        self.cut_width_spin.setValue(self.my_Setting.settings.value('cut_w', type=int))

        self.cut_height_spin = MyQSpinBox()
        self.cut_height_spin.setPrefix('高：')
        self.cut_height_spin.setValue(self.my_Setting.settings.value('cut_h', type=int))
        hbox.addWidget(self.cut_width_spin)
        hbox.addWidget(self.cut_height_spin)
        gridlayout.addWidget(cut_size_lab, 4, 0)
        gridlayout.addLayout(hbox, 4, 1)

        original_epoch_lab = QLabel('原始迭代')
        self.original_epoch_spin = MyQSpinBox()
        add_epoch_lab = QLabel('追加迭代')
        self.add_epoch_spin = MyQSpinBox()
        epoch_hbox.addWidget(original_epoch_lab)
        epoch_hbox.addWidget(self.original_epoch_spin)
        epoch_hbox.addStretch()
        epoch_hbox.addWidget(add_epoch_lab)
        epoch_hbox.addWidget(self.add_epoch_spin)
        gridlayout.addLayout(epoch_hbox, 5, 0, 1, 4)

        batchSize_lab = QLabel('单位批次')
        self.batchSize_num = MyQSpinBox()
        self.batchSize_num.setValue(self.my_Setting.settings.value('batch_size', type=int))
        gridlayout.addWidget(batchSize_lab, 6, 0)
        gridlayout.addWidget(self.batchSize_num, 6, 1)

        save_model_lab = QLabel('保存模型', self)
        self.save_model_line = QLineEdit()
        self.save_model_line.setReadOnly(True)
        save_model_btn = QToolButton()
        save_model_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        save_model_btn.setToolTip('选择模型保存路径')
        gridlayout.addWidget(save_model_lab, 8, 0)
        gridlayout.addWidget(self.save_model_line, 8, 1, 1, 2)
        gridlayout.addWidget(save_model_btn, 8, 3)

        start_btn = QPushButton(QIcon('../Res/VectorEditor_StartEdit.png'), '开始', self)
        stop_btn = QPushButton(QIcon('../Res/VectorEditor_StopEdit.png'), '取消', self)
        btn_hbox.addStretch(1)
        btn_hbox.addWidget(start_btn)
        btn_hbox.addWidget(stop_btn)
        gridlayout.addLayout(btn_hbox, 9, 1)

        self.setLayout(gridlayout)

        """         各个信号         """
        img_path_btn.clicked.connect(lambda : self.select_slot_func('img_path_btn'))
        lab_path_btn.clicked.connect(lambda : self.select_slot_func('lab_path_btn'))
        select_model_btn.clicked.connect(lambda : self.select_slot_func('select_model_btn'))
        save_model_btn.clicked.connect(lambda : self.select_slot_func('save_model_btn'))
        start_btn.clicked.connect(self.start_func)


    def select_slot_func(self, val):
        if val == 'img_path_btn':
            file_dialog = QFileDialog()
            file_path = file_dialog.getExistingDirectory(self, '选择训练图像样本目录')
            self.img_path_line.setText(file_path)

        elif val == 'lab_path_btn':
            file_dialog = QFileDialog()
            file_path = file_dialog.getExistingDirectory(self, '选择训练标签样本目录')
            self.lab_path_line.setText(file_path)

        elif val == 'select_model_btn':
            file_dialog = QFileDialog()
            file_name = file_dialog.getOpenFileName(self, '选择你训练好的模型', './',
                                                    'H5(*.h5);;ALL(*.*)', 'H5(*.h5)')
            self.select_model_line.setText(file_name[0])
        else:
            file_dialog = QFileDialog()
            file_path = file_dialog.getSaveFileName(self, '选择模型保存路径', './',
                                                    'H5 (*.h5)', 'H5 (*.h5)')
            self.save_model_line.setText(file_path[0])

    def start_func(self):
        error_text = ''
        if self.img_path_line.text() == '':
            error_text += '训练图像样本目录为空\n'

        if self.lab_path_line.text() == '':
            error_text += '训练标签样本目录为空\n'

        if self.select_model_line.text() == '':
            error_text += '未选择您要再训练的模型\n'

        if self.save_model_line.text() == '':
            error_text += '保存模型路径为空\n'

        if self.original_epoch_spin.value() < 0:
            error_text += '原始迭代次数小于0\n'

        if self.batchSize_num.value() == 0:
            error_text += 'BatchSize为0\n'

        if self.add_epoch_spin.value() == 0:
            error_text += '增加迭代次数为0\n'

        if self.cut_width_spin.value() == 0 or self.cut_height_spin.value() == 0:
            error_text += '输入图像尺寸过小\n'

        if error_text != '':
            dialog = QErrorMessage()
            dialog.showMessage(error_text)
            dialog.exec()
            return None

        if self.isStarting == False:
            self.isStarting = True
            self.again_train_thread.get_parameters(self.img_path_line.text(), self.lab_path_line.text(),
                                             self.save_model_line.text(), self.original_epoch_spin.value(),
                                             self.batchSize_num.value(), self.cut_width_spin.value(),
                                             self.cut_height_spin.value(), self.add_epoch_spin.value(),
                                             self.select_model_line.text())
            self.again_train_thread.start()

    def stop_func(self):
        self.isStarting = False
        if self.again_train_thread.isRunning():
            try:
                self.again_train_thread.quit()
                self.again_train_thread.terminate()
                self.again_train_thread.wait()
            except Exception as e:
                print('出现错误 --> 错误原因是：' + str(e))

    def my_thread_finished(self):
        self.isStarting = False
        print('完成模型再训练！')

    def save_settings(self):
        self.my_Setting.settings.setValue('batch_size', self.batchSize_num.value())
        self.my_Setting.settings.setValue('cut_w', self.cut_width_spin.value())
        self.my_Setting.settings.setValue('cut_h', self.cut_height_spin.value())
        self.my_Setting.settings.sync()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if self.again_train_thread.isRunning():
            a0.ignore()
            mb = QMessageBox(QMessageBox.Information, '提示', '请点击取消按钮，关闭当前进程后退出', QMessageBox.Ok)
            mb.exec()
        else:
            self.save_settings()
            if self.again_train_thread.isFinished():
                del self.again_train_thread


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Again_Train_Model_Win()
    w.show()
    sys.exit(app.exec())