import sys
from PyQt5 import QtGui
from PyQt5.Qt import QErrorMessage, QLabel, QApplication, QIcon, QGridLayout, QFileDialog, QMessageBox, \
    QRadioButton, QButtonGroup, QPushButton, QToolButton, QHBoxLayout, QWidget, QThread, QLineEdit, \
    QTextEdit
from MySpinBox import MyQSpinBox
# from MyLineEdit import MyQLineEdit
from Algo.SinglePred import Pred_Module
from MySettings import MyQSettings


class Predict_Thread(QThread):
    def __init__(self):
        super(Predict_Thread, self).__init__()

    def run(self):
        try:
            pred_module = Pred_Module()
            pred_module.load_model(self.model_path, self.cut_width)
            if self.mode == 1:  # 单张或者多张预测
                pred_module.imgs_pred(self.img_path, self.save_img_path)
            else:               # 对整个目录中所有的图像预测
                pred_module.dir_pred(self.img_path, self.save_img_path)

        except Exception as e:
            print('错误原因是：' + str(e))

    def get_parameters(self, img_path, model_path, save_img_path, cut_width, cut_height, mode):
        self.img_path = img_path                # 待预测图像
        self.model_path = model_path            # 模型
        self.save_img_path = save_img_path      # 结果保存
        self.cut_width = cut_width              # 图像宽度
        self.cut_height = cut_height            # 图像高度
        self.mode = mode                        # 预测模式->单张、多张、整体预测。


class Predict_Window(QWidget):
    def __init__(self):
        super(Predict_Window, self).__init__()
        self.setWindowTitle('预测界面')
        self.setFixedSize(400, 500)
        self.setContentsMargins(20, 20, 20, 20)

        self.isStarting = False
        self.Mode_FLAG = 0 # 0:初始状态；1：单张预测；2：多张预测；3：整体预测

        self.my_Setting = MyQSettings()
        self.build()
        self.pred_thread = Predict_Thread()
        self.pred_thread.finished.connect(self.Predict_Thread_finished)

    def build(self):
        gridlayout = QGridLayout()
        radio_hbox = QHBoxLayout()

        spin_hbox = QHBoxLayout()
        btn_hbox =QHBoxLayout()

        pred_mode = QLabel('选择预测模式')
        # single_pred_btn = QRadioButton('单张', self)

        multily_pred_btn = QRadioButton('多张预测', self)
        multily_pred_btn.setChecked(True)
        all_pred_btn = QRadioButton('整体预测', self)

        pred_btn_group = QButtonGroup(self)
        pred_btn_group.addButton(multily_pred_btn)
        pred_btn_group.addButton(all_pred_btn)

        radio_hbox.addStretch()
        radio_hbox.addWidget(multily_pred_btn)
        radio_hbox.addStretch()
        radio_hbox.addWidget(all_pred_btn)

        gridlayout.addWidget(pred_mode, 1, 0)
        gridlayout.addLayout(radio_hbox, 1, 1, 1, 2)


        pred_img_lab = QLabel('选择预测图像', self)
        self.pred_img_line = QLineEdit()
        self.pred_img_line.setReadOnly(True)

        pred_img_btn = QToolButton(self)
        pred_img_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        pred_img_btn.setToolTip('选择预测图像')
        gridlayout.addWidget(pred_img_lab, 2, 0)
        gridlayout.addWidget(self.pred_img_line, 2, 1, 1, 2)
        gridlayout.addWidget(pred_img_btn, 2, 3)

        select_model_lab = QLabel('选择预测模型')
        self.select_model_line = QLineEdit()
        self.select_model_line.setReadOnly(True)
        select_model_btn = QToolButton(self)
        select_model_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        select_model_btn.setToolTip('选择模型')
        gridlayout.addWidget(select_model_lab, 3, 0)
        gridlayout.addWidget(self.select_model_line, 3, 1, 1, 2)
        gridlayout.addWidget(select_model_btn, 3, 3)

        in_size_model_lab = QLabel('模型输入尺寸')
        self.sb_w = MyQSpinBox()
        self.sb_w.setValue(self.my_Setting.settings.value('cut_w', type=int))
        self.sb_w.setPrefix('宽：')
        self.sb_h = MyQSpinBox()
        self.sb_h.setValue(self.my_Setting.settings.value('cut_h', type=int))
        self.sb_h.setPrefix('高：')
        spin_hbox.addWidget(self.sb_w)
        spin_hbox.addWidget(self.sb_h)
        gridlayout.addWidget(in_size_model_lab, 4, 0)
        gridlayout.addLayout(spin_hbox, 4, 1, 1, 2)

        save_result_lab = QLabel('保存预测结果')
        self.save_result_line = QLineEdit()
        self.save_result_line.setReadOnly(True)
        save_result_btn = QToolButton()
        save_result_btn.setIcon(QIcon('../Res/SIAS_Open.png'))
        save_result_btn.setToolTip('保存预测结果')
        gridlayout.addWidget(save_result_lab, 5, 0)
        gridlayout.addWidget(self.save_result_line, 5, 1, 1, 2)
        gridlayout.addWidget(save_result_btn, 5, 3)

        start_btn = QPushButton(QIcon('../Res/VectorEditor_StartEdit.png'), '开始', self)
        stop_btn = QPushButton(QIcon('../Res/VectorEditor_StopEdit.png'), '取消', self)
        btn_hbox.addStretch(1)
        btn_hbox.addWidget(start_btn)
        btn_hbox.addWidget(stop_btn)
        gridlayout.addLayout(btn_hbox, 6, 1)

        self.setLayout(gridlayout)

        self.pred_btn_group = pred_btn_group
        pred_btn_group.buttonToggled.connect(self.pred_btn_group_func)
        pred_img_btn.clicked.connect(self.select_img_path_func)
        select_model_btn.clicked.connect(self.select_model_func)
        save_result_btn.clicked.connect(self.save_result_func)
        stop_btn.clicked.connect(self.stop_func)
        start_btn.clicked.connect(self.start_func)

    def pred_btn_group_func(self, val):
        # print(self.pred_btn_group.id(val))
        self.pred_img_line.clear()
        self.save_result_line.clear()

    def select_img_path_func(self):
        dialog = QFileDialog()
        # if self.pred_btn_group.checkedButton().text() == '单张':
        #     file_name = dialog.getOpenFileName(self, '选择待预测图像', './',
        #                 'TIF(*.tif *.tiff);;PNG(*.png);;JPEG(*.jpg);;ALL(*.*)', 'TIF(*.tif *.tiff)')
        #     self.pred_img_line.setText(file_name[0])
        #     self.Mode_FLAG = 1
        #     # print(file_name)

        if self.pred_btn_group.checkedButton().text() == '多张预测':
            file_names = dialog.getOpenFileNames(self, '选择待预测图像', './',
                        'TIF(*.tif *.tiff);;PNG(*.png);;JPEG(*.jpg);;ALL(*.*)', 'TIF(*.tif *.tiff)')
            # print(file_names)
            names = ''
            for i in file_names[0]:
                names += i + ';'
            self.pred_img_line.setText(names)
            self.Mode_FLAG = 1
        else:
            file_path = dialog.getExistingDirectory(self, '选择待预测图像所在目录', './')
            self.pred_img_line.setText(file_path)
            self.Mode_FLAG = 2
            # print(file_path)

    def select_model_func(self):
        dialog = QFileDialog()
        file_name = dialog.getOpenFileName(self, '选择预测模型', './', 'H5(*.h5)', 'H5(*.h5)')
        self.select_model_line.setText(file_name[0])

    def save_result_func(self):
        dialog = QFileDialog()
        # if self.pred_btn_group.checkedButton().text() == '单张':
        #     file_name = dialog.getSaveFileName(self, '保存预测结果', './',
        #             'TIF(*.tif *.tiff);;PNG(*.png);;JPEG(*.jpg);;ALL(*.*)', 'TIF(*.tif *.tiff)')
        #     self.save_result_line.setText(file_name[0])
        # else:
        file_path = dialog.getExistingDirectory(self, '选择结果保存目录', './')
        self.save_result_line.setText(file_path)

    def start_func(self):
        error_text = ''
        if self.pred_img_line.text() == '':
            error_text += '预测图像为空\n'
        if self.select_model_line.text() == '':
            error_text += '未加载模型\n'
        if self.save_result_line.text() == '':
            error_text += '结果保存目录为空\n'
        if self.sb_h == 0 or self.sb_w == 0:
            error_text += '模型输入尺寸为0\n'

        if error_text != '':
            dialog = QErrorMessage(self)
            dialog.showMessage(error_text)
            return None

        if self.isStarting == False:
            self.isStarting = True
            self.pred_thread.get_parameters(
                img_path=self.pred_img_line.text(),
                model_path=self.select_model_line.text(),
                save_img_path=self.save_result_line.text(),
                cut_width=self.sb_w.value(),
                cut_height=self.sb_h.value(),
                mode = self.Mode_FLAG
            )
            self.pred_thread.start()

    def stop_func(self):
        self.isStarting = False
        if self.pred_thread.isRunning():
            try:
                self.pred_thread.quit()
                self.pred_thread.terminate()
                self.pred_thread.wait()
            except Exception as e:
                print('出现错误 --> 错误原因是：' + str(e))

    def Predict_Thread_finished(self):
        self.isStarting = False
        print('完成图像预测！')

    def save_settings(self):
        self.my_Setting.settings.setValue('cut_w', self.sb_w.value())
        self.my_Setting.settings.setValue('cut_h', self.sb_h.value())
        self.my_Setting.settings.sync()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if self.pred_thread.isRunning():
            a0.ignore()
            mb = QMessageBox(QMessageBox.Information, '提示', '请点击取消按钮，关闭当前进程后退出', QMessageBox.Ok)
            mb.exec()
        else:
            self.save_settings()
            if self.pred_thread.isFinished():
                del self.pred_thread

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pred = Predict_Window()
    pred.show()
    sys.exit(app.exec())
