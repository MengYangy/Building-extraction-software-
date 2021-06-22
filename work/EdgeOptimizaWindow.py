import sys
from PyQt5 import QtGui
from PyQt5.Qt import QErrorMessage, QLabel, QApplication, QIcon, QGridLayout, QFileDialog, QMessageBox, \
    QRadioButton, QButtonGroup, QPushButton, QToolButton, QHBoxLayout, QWidget, QThread, QLineEdit, QDoubleSpinBox
from MySpinBox import MyQSpinBox
from MySettings import MyQSettings
from MyPrint import myPrint


class Edge_OP_Thread(QThread):
    def __init__(self):
        super(Edge_OP_Thread, self).__init__()

    def run(self) -> None:
        try:

            if self.mode == 1:  # 单张检测
               myPrint(1)

            else:               # 多张检测
                myPrint(2)
        except Exception as e:
            print('错误原因是：' + str(e))

    def get_parameters(self, img_path, save_img_path, isGenerator, mode):
        self.img_path = img_path            # 待检测图像
        self.save_img_path = save_img_path  # 结果保存
        self.isGenerator = isGenerator      # 是否生产矢量图
        self.mode = mode                    # 预测模式->单张、多张、整体预测。


class Edge_Optimiza_Win(QWidget):
    def __init__(self):
        super(Edge_Optimiza_Win, self).__init__()
        self.setWindowTitle('边缘优化界面')
        self.setFixedSize(400, 500)
        self.setContentsMargins(20, 20, 20, 20)
        self.isStarting = False
        self.isGenerate = True
        self.Mode_FLAGE = 0

        self.edge_op_thread = Edge_OP_Thread()
        self.edge_op_thread.finished.connect(self.edge_op_thread_finished_func)

        self.build()

    def build(self):
        print('--------')
        gridLayout = QGridLayout()
        radio_hbox = QHBoxLayout()
        yesORno_hbox = QHBoxLayout()
        btn_hbox = QHBoxLayout()

        detect_mode = QLabel('选择优化模式')
        multily_detect_btn = QRadioButton('多张优化', self)
        multily_detect_btn.setChecked(True)
        all_detect_btn = QRadioButton('整体优化', self)

        optimiza_btn_group = QButtonGroup(self)
        optimiza_btn_group.addButton(multily_detect_btn)
        optimiza_btn_group.addButton(all_detect_btn)

        radio_hbox.addStretch()
        radio_hbox.addWidget(multily_detect_btn)
        radio_hbox.addStretch()
        radio_hbox.addWidget(all_detect_btn)

        gridLayout.addWidget(detect_mode, 1, 0)
        gridLayout.addLayout(radio_hbox, 1, 1, 1, 2)

        optimiza_img_path_lab = QLabel('加载优化数据')
        self.optimiza_img_path_line = QLineEdit()
        self.optimiza_img_path_line.setReadOnly(True)
        optimiza_img_path_btn = QToolButton()

        optimiza_img_path_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        optimiza_img_path_btn.setToolTip('选择原始图像')
        gridLayout.addWidget(optimiza_img_path_lab, 2, 0)
        gridLayout.addWidget(self.optimiza_img_path_line, 2, 1, 1, 2)
        gridLayout.addWidget(optimiza_img_path_btn, 2, 3)

        optimiza_result_path_lab = QLabel('保存优化结果')
        self.optimiza_result_path_line = QLineEdit()
        self.optimiza_result_path_line.setReadOnly(True)
        optimiza_result_path_btn = QToolButton()
        optimiza_result_path_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        optimiza_result_path_btn.setToolTip('保存优化结果')
        gridLayout.addWidget(optimiza_result_path_lab, 3, 0)
        gridLayout.addWidget(self.optimiza_result_path_line, 3, 1, 1, 2)
        gridLayout.addWidget(optimiza_result_path_btn, 3, 3)

        generate_vector_data = QLabel('生成矢量数据')
        yes_radio = QRadioButton('是', self)
        yes_radio.setChecked(True)
        no_radio = QRadioButton('否', self)

        yesORno_btn_group = QButtonGroup(self)
        yesORno_btn_group.addButton(yes_radio)
        yesORno_btn_group.addButton(no_radio)
        yesORno_hbox.addStretch()
        yesORno_hbox.addWidget(yes_radio)
        yesORno_hbox.addStretch()
        yesORno_hbox.addWidget(no_radio)

        gridLayout.addWidget(generate_vector_data, 4, 0)
        gridLayout.addLayout(yesORno_hbox, 4, 1, 1, 2)

        start_btn = QPushButton(QIcon('../Res/VectorEditor_StartEdit.png'), '开始', self)
        stop_btn = QPushButton(QIcon('../Res/VectorEditor_StopEdit.png'), '取消', self)
        btn_hbox.addStretch(1)
        btn_hbox.addWidget(start_btn)
        btn_hbox.addWidget(stop_btn)
        gridLayout.addLayout(btn_hbox, 5, 1)

        self.setLayout(gridLayout)
        self.optimiza_btn_group = optimiza_btn_group
        self.yesORno_btn_group = yesORno_btn_group
        optimiza_btn_group.buttonToggled.connect(self.optimiza_btn_group_func)
        yesORno_btn_group.buttonToggled.connect(self.yesORno_btn_group_func)
        optimiza_img_path_btn.clicked.connect(lambda : self.select_img_path_func(self.optimiza_img_path_line))
        optimiza_result_path_btn.clicked.connect(self.save_result_path_func)
        start_btn.clicked.connect(self.start_func)
        stop_btn.clicked.connect(self.stop_func)

    def select_img_path_func(self, widget):
        fileDialog = QFileDialog()
        if self.optimiza_btn_group.checkedButton().text() == '多张优化':
            file_names = fileDialog.getOpenFileNames(self, '选择待优化图像', './',
                                                 'TIF(*.tif *.tiff);;PNG(*.png);;JPEG(*.jpg);;ALL(*.*)',
                                                 'TIF(*.tif *.tiff)')
            names = ''
            for i in file_names[0]:
                names += i + ';'
            widget.setText(names)
            self.Mode_FLAGE = 1
        else:
            file_path = fileDialog.getExistingDirectory(self, '选择待预测图像所在目录', './')
            widget.setText(file_path)
            self.Mode_FLAGE = 2

    def save_result_path_func(self):
        fileDialog = QFileDialog()
        file_path = fileDialog.getExistingDirectory(self, '选择待预测图像所在目录', './')
        self.optimiza_result_path_line.setText(file_path)

    def start_func(self):
        error_text = ''
        print(error_text)
        if self.optimiza_img_path_line.text() == '':
            error_text += '待优化图像为空\n'
        if self.optimiza_result_path_line.text() == '':
            error_text += '结果保存目录为空\n'
        if error_text != '':
            dialog = QErrorMessage(self)
            dialog.showMessage(error_text)
            return None

        if self.isStarting == False:
            self.isStarting = True
            self.edge_op_thread.get_parameters(
                img_path = self.optimiza_img_path_line.text(),
                save_img_path = self.optimiza_result_path_line.text(),
                mode = self.Mode_FLAGE,
                isGenerator=self.isGenerate
            )
            self.edge_op_thread.start()

    def stop_func(self):
        self.isStarting = False
        if self.edge_op_thread.isRunning():
            try:
                self.edge_op_thread.quit()
                self.edge_op_thread.terminate()
                self.edge_op_thread.wait()
            except Exception as e:
                print('出现错误 --> 错误原因是：' + str(e))

    def edge_op_thread_finished_func(self):
        self.isStarting = False
        print('完成当前边缘优化任务')

    def optimiza_btn_group_func(self):
        self.optimiza_img_path_line.clear()
        self.optimiza_result_path_line.clear()

    def yesORno_btn_group_func(self):
        if self.yesORno_btn_group.checkedButton().text() == '是':
            self.isGenerate = True
        else:
            self.isGenerate = False
        # print(self.isGenerate)


    def save_settings(self):
        pass

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if self.edge_op_thread.isRunning():
            a0.ignore()
            mb = QMessageBox(QMessageBox.Information, '提示', '请点击取消按钮，关闭当前进程后退出', QMessageBox.Ok)
            mb.exec()
        else:
            self.save_settings()
            if self.edge_op_thread.isFinished():
                del self.edge_op_thread


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Edge_Optimiza_Win()
    w.show()
    sys.exit(app.exec())