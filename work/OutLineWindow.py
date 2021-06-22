import sys
from PyQt5 import QtGui
from PyQt5.Qt import QErrorMessage, QLabel, QApplication, QIcon, QGridLayout, QFileDialog, QMessageBox, \
    QRadioButton, QButtonGroup, QPushButton, QToolButton, QHBoxLayout, QWidget, QThread, QLineEdit, QDoubleSpinBox
from MySpinBox import MyQSpinBox
from MySettings import MyQSettings
from Algo.SingleArea import Contour_Detection


class OutLine_Thread(QThread):
    def __init__(self):
        super(OutLine_Thread, self).__init__()

    def run(self):
        try:
            # myPrint(1)
            '''
            mode                1           2
            img_path:       多张图像        路径
            pred_path:      多张图像        路径
            save_img_path:  路径            路径
            resolution:     float           float
            '''
            contour_Detection = Contour_Detection()
            if self.mode == 1:  # 单张检测
                contour_Detection.img_detection(imgs=self.img_path,
                                                pred_imgs=self.pred_path,
                                                save_path=self.save_img_path,
                                                resolution=self.resolution)
            else:               # 多张检测
                contour_Detection.dir_detection(imgs=self.img_path,
                                                pred_imgs=self.pred_path,
                                                save_path=self.save_img_path,
                                                resolution=self.resolution)
        except Exception as e:
            print('错误原因是：' + str(e))

    def get_parameters(self, img_path, pred_path, save_img_path, resolution, mode):
        self.img_path = img_path            # 待检测图像
        self.pred_path = pred_path          # 已预测结果
        self.save_img_path = save_img_path  # 结果保存
        self.resolution = resolution        # 空间分辨率
        self.mode = mode                    # 预测模式->单张、多张、整体预测。

class OutLine_Window(QWidget):
    def __init__(self):
        super(OutLine_Window, self).__init__()
        self.setWindowTitle('轮廓检测界面')
        self.setFixedSize(400, 500)
        self.setContentsMargins(20, 20, 20, 20)

        self.my_Setting = MyQSettings()
        self.build()
        self.isStarting = False
        self.Mode_FLAGE = 0

        self.outline_thread = OutLine_Thread()
        self.outline_thread.finished.connect(self.outline_thread_finished_func)

    def build(self):
        gridLayout = QGridLayout()
        radio_hbox = QHBoxLayout()
        btn_hbox = QHBoxLayout()

        detect_mode = QLabel('选择检测模式')
        multily_detect_btn = QRadioButton('多张检测', self)
        multily_detect_btn.setChecked(True)
        all_detect_btn = QRadioButton('整体检测', self)

        detect_btn_group = QButtonGroup(self)
        detect_btn_group.addButton(multily_detect_btn)
        detect_btn_group.addButton(all_detect_btn)

        radio_hbox.addStretch()
        radio_hbox.addWidget(multily_detect_btn)
        radio_hbox.addStretch()
        radio_hbox.addWidget(all_detect_btn)

        gridLayout.addWidget(detect_mode, 1, 0)
        gridLayout.addLayout(radio_hbox, 1, 1, 1, 2)

        img_path_lab = QLabel('选择原始图像', self)
        self.img_path_line = QLineEdit()
        self.img_path_line.setReadOnly(True)
        img_path_btn = QToolButton(self)
        img_path_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        img_path_btn.setToolTip('选择原始图像')
        gridLayout.addWidget(img_path_lab, 2, 0)
        gridLayout.addWidget(self.img_path_line, 2, 1, 1, 2)
        gridLayout.addWidget(img_path_btn, 2, 3)

        pred_img_lab = QLabel('选择预测结果')
        self.pred_img_line = QLineEdit()
        self.pred_img_line.setReadOnly(True)
        pred_img_btn = QToolButton(self)
        pred_img_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        pred_img_btn.setToolTip('选择预测结果')
        gridLayout.addWidget(pred_img_lab, 3, 0)
        gridLayout.addWidget(self.pred_img_line, 3, 1, 1, 2)
        gridLayout.addWidget(pred_img_btn, 3, 3)

        save_result_lab = QLabel('保存检测结果')
        self.save_result_line = QLineEdit()
        self.save_result_line.setReadOnly(True)
        save_result_btn = QToolButton()
        save_result_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        save_result_btn.setToolTip('保存检测结果')
        gridLayout.addWidget(save_result_lab, 4, 0)
        gridLayout.addWidget(self.save_result_line, 4, 1, 1, 2)
        gridLayout.addWidget(save_result_btn, 4, 3)

        resolution_lab = QLabel('空间分辨率为')
        self.resolution_spin = QDoubleSpinBox(self)
        self.resolution_spin.setDecimals(3)
        self.resolution_spin.setSingleStep(0.01)
        self.resolution_spin.setMaximumWidth(80)
        self.resolution_spin.setValue(self.my_Setting.settings.value('resolution', type=float))
        gridLayout.addWidget(resolution_lab, 5, 0)
        gridLayout.addWidget(self.resolution_spin, 5, 1, 1, 1)

        start_btn = QPushButton(QIcon('../Res/VectorEditor_StartEdit.png'), '开始', self)
        stop_btn = QPushButton(QIcon('../Res/VectorEditor_StopEdit.png'), '取消', self)
        btn_hbox.addStretch(1)
        btn_hbox.addWidget(start_btn)
        btn_hbox.addWidget(stop_btn)
        gridLayout.addLayout(btn_hbox, 6, 1)

        self.setLayout(gridLayout)
        self.detect_btn_group = detect_btn_group
        detect_btn_group.buttonToggled.connect(self.detect_btn_group_func)
        img_path_btn.clicked.connect(lambda : self.select_img_path_func(self.img_path_line))
        pred_img_btn.clicked.connect(lambda : self.select_img_path_func(self.pred_img_line))
        save_result_btn.clicked.connect(self.save_result_func)
        stop_btn.clicked.connect(self.stop_func)
        start_btn.clicked.connect(self.start_func)

    def outline_thread_finished_func(self):
        self.isStarting = False
        print('完成轮廓检测！')

    def detect_btn_group_func(self, val):
        self.img_path_line.clear()
        self.pred_img_line.clear()
        self.save_result_line.clear()

    def select_img_path_func(self, widget):
        dialog = QFileDialog()
        if self.detect_btn_group.checkedButton().text() == '多张检测':
            file_names = dialog.getOpenFileNames(self, '选择待预测图像', './',
                        'TIF(*.tif *.tiff);;PNG(*.png);;JPEG(*.jpg);;ALL(*.*)', 'TIF(*.tif *.tiff)')
            names = ''
            for i in file_names[0]:
                names += i + ';'
            widget.setText(names)
            self.Mode_FLAGE = 1
        else:
            file_path = dialog.getExistingDirectory(self, '选择待预测图像所在目录', './')
            widget.setText(file_path)
            self.Mode_FLAGE = 2

    def save_result_func(self):
        dialog = QFileDialog()

        file_path = dialog.getExistingDirectory(self, '选择结果保存目录', './')
        self.save_result_line.setText(file_path)

    def start_func(self):
        error_text = ''
        if self.img_path_line == '':
            error_text += '原始图像为空\n'
        if self.pred_img_line == '':
            error_text += '预测图像为空\n'
        if self.save_result_line == '':
            error_text += '结果保存目录为空\n'
        if error_text != '':
            dialog = QErrorMessage(self)
            dialog.showMessage(error_text)
            return None

        if self.isStarting == False and self.outline_thread.isRunning() == False:
            self.isStarting = True
            self.outline_thread.get_parameters(
                img_path = self.img_path_line.text(),
                pred_path = self.pred_img_line.text(),
                save_img_path = self.save_result_line.text(),
                mode = self.Mode_FLAGE,
                resolution=self.resolution_spin.value()
            )
            self.outline_thread.start()

    def stop_func(self):
        self.isStarting = False
        if self.outline_thread.isRunning():
            try:
                self.outline_thread.quit()
                self.outline_thread.terminate()
                self.outline_thread.wait()
            except Exception as e:
                print('出现错误 --> 错误原因是：' + str(e))


    def save_settings(self):
        self.my_Setting.settings.setValue('resolution', self.resolution_spin.value())
        self.my_Setting.settings.sync()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if self.outline_thread.isRunning():
            a0.ignore()
            mb = QMessageBox(QMessageBox.Information, '提示', '请点击取消按钮，关闭当前进程后退出', QMessageBox.Ok)
            mb.exec()
        else:
            self.save_settings()
            if self.outline_thread.isFinished():
                del self.outline_thread


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = OutLine_Window()
    w.show()
    sys.exit(app.exec())