from PyQt5.Qt import QMainWindow, QDockWidget, QLabel, Qt, QApplication, QAction, QIcon, pyqtSignal, \
    QTreeWidget, QTreeWidgetItem, QFileDialog, QImage, QPixmap, QGraphicsView, QGraphicsPixmapItem, \
    QCursor, QSettings, QToolButton, QHBoxLayout, QThread, QTextEdit, QPlainTextEdit, QMessageBox
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QObject
import sys, os
import cv2 as cv
import time
from work.DataProceWindow import Data_pro_win
from TrainModelWindow import Train_Model_Win
from AgainTrainWindow import Again_Train_Model_Win
from PredictWindow import Predict_Window
from OutLineWindow import OutLine_Window
from EdgeOptimizaWindow import Edge_Optimiza_Win


class Emit_Str(QObject):
    text_reWrite = pyqtSignal(str)
    def write(self, text):
        self.text_reWrite.emit(str(text))

class MyLabel(QLabel):
    def __init__(self, parent=None):
        super(MyLabel, self).__init__(parent)
        self.setStyleSheet('background-color:rgb(255,255,255)')
        self.setMouseTracking(True)     # 打开监听鼠标事件
        self.setMargin(10)
        self.setScaledContents(True)
        self.grabKeyboard()
        self.isClick = False
        self.isMove = False
        self.isMoveBtn = False
        self.my_child = None
        self.my_child_x = 0
        self.my_child_y = 0
        self.is_Shift = False
        self.is_ctrl = False

    itMove = pyqtSignal(float, float)
    itClick = pyqtSignal(int, int)
    up_Wheel =pyqtSignal()
    down_Wheel = pyqtSignal()
    left_Wheel = pyqtSignal()
    right_Wheel = pyqtSignal()
    ctrl_zoom_out = pyqtSignal()
    ctrl_zoom_in = pyqtSignal()
    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.itMove.emit(ev.x(), ev.y())
        if self.isClick:
            self.isMove = True

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.itClick.emit(ev.x(), ev.y())
        if ev.button() == Qt.MidButton: #  or (self.isMoveBtn and ev.button() == Qt.LeftButton)
            # print('self.isMoveBtn',self.isMoveBtn)
            # print('ev.button() == Qt.LeftButton', ev.button() == Qt.LeftButton)
            self.setCursor(Qt.ClosedHandCursor)
            self.isClick = True
            if self.my_child != None:
                # print(self.my_child.x(),self.my_child.y())
                self.my_child_x = self.my_child.x()
                self.my_child_y = self.my_child.y()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.isClick = False
        self.isMove = False
        self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, a0: QtGui.QWheelEvent) -> None:
        if self.is_Shift:
            if a0.angleDelta().y() > 0:
                # print('向上', a0.pos())
                self.left_Wheel.emit()
            else:
                # print('向下', a0.pos())
                self.right_Wheel.emit()
        elif self.is_ctrl:
            if a0.angleDelta().y() > 0:
                self.ctrl_zoom_out.emit()
            else:
                print('zoom in')
                self.ctrl_zoom_in.emit()
        else:
            if a0.angleDelta().y() > 0:
                # print('向上', a0.pos())
                self.up_Wheel.emit()
            else:
                # print('向下', a0.pos())
                self.down_Wheel.emit()


    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        # print('键盘事件')
        if ev.key() == Qt.Key_Shift:
            self.is_Shift = True

        if ev.key() == Qt.Key_Control:
            self.is_ctrl = True
            cursor = QCursor(QPixmap('../Res/Identify_RasterIdentify.png'))
            self.setCursor(cursor)


    def keyReleaseEvent(self, a0: QtGui.QKeyEvent) -> None:
        # print('松开键盘事件')
        if a0.key() == Qt.Key_Shift:
            self.is_Shift = False

        if a0.key() == Qt.Key_Control:
            self.is_ctrl = False
            self.setCursor(Qt.ArrowCursor)

    def enterEvent(self, a0: QtCore.QEvent) -> None:
        self.grabKeyboard()
        QLabel.enterEvent(self, a0)

    def leaveEvent(self, a0: QtCore.QEvent) -> None:
        self.releaseKeyboard()
        QLabel.leaveEvent(self, a0)

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        print(self.size())



class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle('建筑物提取平台')
        self.resize(1280, 720)
        self.setMinimumSize(1000, 700)
        self.setContentsMargins(8,0,4,0)
        self.main_Lab = MyLabel()
        self.setCentralWidget(self.main_Lab)
        self.layers_list = []           # 图层列表
        self.layer = None               # 当前显示层
        self.setWindowState(Qt.WindowMaximized)
        self.move_speed = 30            # 图片移动速度
        self.data_proce_isClick = False
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)


        self.add_menu()
        self.add_tool()
        self.add_func_tool()
        self.add_status()
        self.add_dock()
        self.add_output_dock()
        self.signal_action()

        sys.stdout = Emit_Str(text_reWrite=self.output_text_write)
        sys.stderr = Emit_Str(text_reWrite=self.output_text_write)
        # self.create_ini()

    def output_text_write(self, text):
        if text != '\n':
            self.output_text.appendPlainText(text)

    def create_ini(self):
        settings = QSettings('./config.ini', QSettings.IniFormat)
        settings.setValue('epoch_num', 50)
        settings.setValue('batch_size', 6)
        settings.setValue('class_num', 2)
        # settings.setValue('resolution', '0.01')
        settings.sync()
        # print(settings.value('cut_num'))
        # print(type(settings.value('cut_num')))

    def add_menu(self):
        self.main_menu = self.menuBar()
        # 文件菜单
        file = self.main_menu.addMenu('文件')
        new_file = QAction('新建', self.main_menu)
        file.addAction(new_file)
        new_file.triggered.connect(lambda : print('点击了新建按钮'))

        open_file = QAction('打开', self.main_menu)
        file.addAction(open_file)
        open_file.triggered.connect(lambda: print('点击了打开按钮'))

        file.addSeparator()

        exit_file = QAction('退出', self.main_menu)
        file.addAction(exit_file)
        exit_file.triggered.connect(lambda : print('点击了退出按钮')) # self.close

        # 编辑菜单
        edit = self.main_menu.addMenu('编辑')
        start_edit = QAction('开始编辑', self.main_menu)
        edit.addAction(start_edit)
        start_edit.triggered.connect(lambda: print('点击了开始编辑按钮'))

        save_edit = QAction('保存编辑', self.main_menu)
        edit.addAction(save_edit)
        save_edit.triggered.connect(lambda: print('点击了保存编辑按钮'))

        edit.addSeparator()

        stop_edit = QAction('结束编辑', self.main_menu)
        edit.addAction(stop_edit)
        stop_edit.triggered.connect(lambda: print('点击了结束编辑按钮'))
        # 帮助菜单
        help = self.main_menu.addMenu('帮助')
        help_action = QAction('帮助', self.main_menu)
        help.addAction(help_action)
        help_action.triggered.connect(lambda: print('点击了帮助按钮'))

        about_action = QAction('关于', self.main_menu)
        help.addAction(about_action)
        about_action.triggered.connect(lambda: print('点击了关于按钮'))

    def add_tool(self):
        tool = self.addToolBar('工具栏')
        tool.setToolTip('工具栏')
        self.open_img_action = QAction(QIcon('../Res/ui_ImageMosaic.png'), '打开', self)
        tool.addAction(self.open_img_action)

        self.zoom_img_action = QAction(QIcon('../Res/CartoGraphy_ZoomIn.png'), '居中放大', self)
        tool.addAction(self.zoom_img_action)

        self.lessen_img_action = QAction(QIcon('../Res/CartoGraphy_ZoomOut.png'), '居中缩小', self)
        tool.addAction(self.lessen_img_action)

        self.select_action = QAction(QIcon('../Res/MapBrowser_SelectRaster.png'), '指针', self)
        tool.addAction(self.select_action)

        self.move_img_action = QAction(QIcon('../Res/MapBrowser_Pan.png'), '移动', self)
        tool.addAction(self.move_img_action)

        self.toPrevious_action = QAction(QIcon('../Res/CartoGraphy_ZoomToPreviousExtent.png'), '上一张', self)
        tool.addAction(self.toPrevious_action)

        self.toNext_action = QAction(QIcon('../Res/CartoGraphy_ZoomToNextExtent.png'), '下一张', self)
        tool.addAction(self.toNext_action)

        self.add_layer_action = QAction(QIcon('../Res/CloudFlash_LayerCollection.png'), '添加图层', self)
        tool.addAction(self.add_layer_action)
        tool.setMinimumWidth(400)

    def add_func_tool(self):
        func_tool = self.addToolBar('功能工具栏')
        func_tool.setToolTip('功能工具栏')
        self.data_processing_action = QAction(QIcon('../Res/DeepLearning_Building.png'), '数据增强', self)
        func_tool.addAction(self.data_processing_action)

        self.train_model_action = QAction(QIcon('../Res/DeepLearning_Airport.png'), '训练模型', self)
        func_tool.addAction(self.train_model_action)

        self.re_train_model_action = QAction(QIcon('../Res/DeepLearning_Tower.png'),
                                             '模型再训练', self)
        func_tool.addAction(self.re_train_model_action)

        self.predict_action = QAction(QIcon('../Res/DeepLearning_Grass.png'), '预测', self)
        func_tool.addAction(self.predict_action)

        self.outline_action = QAction(QIcon('../Res/MarkTool_DrawNeatline.png'), '轮廓检测', self)
        func_tool.addAction(self.outline_action)

        self.edge_optimization_action = QAction(QIcon('../Res/SIAS_Simply.png'), '边缘优化', self)
        func_tool.addAction(self.edge_optimization_action)

    def add_status(self):
        self.status = self.statusBar()

        self.history_btn = QToolButton(self.status)
        self.history_btn.setIcon(QIcon('../Res/Batch_MultiLook.png'))
        self.history_btn.setToolTip('查看历史记录')
        self.status.addWidget(self.history_btn)

        self.stretch_lab = QLabel()
        self.status.addWidget(self.stretch_lab, 1)

        self.pos_lab = QLabel()
        self.pos_lab.setMinimumWidth(160)
        self.pos_lab.setToolTip('当前坐标')
        self.status.addWidget(self.pos_lab)

    def add_dock(self):
        self.dock = QDockWidget('图层控制器', self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock)
        self.add_tree_to_dock()

    def add_output_dock(self):
        self.out_dock = QDockWidget('输出交互窗口', self)
        # self.out_dock.hide()
        self.addDockWidget(Qt.BottomDockWidgetArea, self.out_dock)
        self.output_text = QPlainTextEdit(self.out_dock)
        self.output_text.setContentsMargins(50,10,10,10)
        self.output_text.setReadOnly(True)
        self.out_dock.setWidget(self.output_text)
        # self.out_dock.setl

    def add_tree_to_dock(self):
        self.tree = QTreeWidget(self.dock)
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(['image','path'])
        self.tree.setHeaderHidden(True)     # 隐藏表头
        self.tree.hideColumn(1)             # 隐藏第2列
        self.tree.setColumnWidth(0, 100)
        self.dock.setWidget(self.tree)
        self.root = QTreeWidgetItem(self.tree)
        self.root.setText(0, 'layer')
        # self.setStyleSheet('QTreeWidget::QTreeWidgetItem:selected{color:red}')

    def add_path_to_tree(self, path_name):
        root = QTreeWidgetItem(self.root)
        root.setText(0, path_name.split('/')[-1])
        root.setText(1, path_name)
        root.setCheckState(0, Qt.Checked)
        self.tree.setCurrentItem(root)

    def signal_action(self):
        self.open_img_action.triggered.connect(self.open_img_func)      # 打开图片信号
        self.main_Lab.itMove.connect(self.update_status_pos)            # 在用户界面鼠标移动信号
        self.main_Lab.itClick.connect(self.mouse_click)
        self.zoom_img_action.triggered.connect(self.zoom_img_func)      # 放大图片信号
        # self.add_layer_action.triggered.connect(self.create_layer)      # 创建图层信号
        self.lessen_img_action.triggered.connect(self.lessen_img_func)  # 缩小图片信号
        self.move_img_action.triggered.connect(self.move_img_btn_func)      # 移动图片信号  self.move_img_func
        self.select_action.triggered.connect(self.select_btn_func)      # 选择类型鼠标
        self.main_Lab.up_Wheel.connect(self.up_wheel_func)              # 滑轮向上事件
        self.main_Lab.down_Wheel.connect(self.down_wheel_func)          # 滑轮向下事件
        self.main_Lab.left_Wheel.connect(self.left_wheel_func)          # 左移事件
        self.main_Lab.right_Wheel.connect(self.right_wheel_func)        # 右移事件
        self.main_Lab.ctrl_zoom_in.connect(self.ctrl_zoom_in_func)      # ctrl 放大事件
        self.main_Lab.ctrl_zoom_out.connect(self.ctrl_zoom_out_func)  # ctrl 放大事件
        # self.tree.clicked.connect(self.toggle_show)
        self.tree.pressed.connect(self.current_tree_change)             # 树控件当前选中发生变化信号
        self.data_processing_action.triggered.connect(self.data_proce_func)      # 打开数据增强界面
        self.train_model_action.triggered.connect(self.train_model_func)    # 打开训练模型界面
        self.toPrevious_action.triggered.connect(self.toPrevious_func)  # 切换至上一张图片
        self.re_train_model_action.triggered.connect(self.again_model_func) # 打开模型再训练界面
        self.predict_action.triggered.connect(self.predict_func)            # 打开预测界面
        self.outline_action.triggered.connect(self.outline_func)            # 打开轮廓检测界面
        self.edge_optimization_action.triggered.connect(self.edge_optimization_func)    # 打开边缘优化界面

    def data_proce_func(self):
        try:
            self.dpw = Data_pro_win()
            self.dpw.show()
        except Exception as e:
            print('出现错误 --> 错误原因是：' + str(e))

    def train_model_func(self):
        try:
            self.train_model = Train_Model_Win()
            self.train_model.show()
        except Exception as e:
            print('出现错误 --> 错误原因是：' + str(e))

    def again_model_func(self):
        try:
            self.again_train = Again_Train_Model_Win()
            self.again_train.show()
        except Exception as e:
            print('出现错误 --> 错误原因是：' + str(e))

    def predict_func(self):
        try:
            self.pred = Predict_Window()
            self.pred.show()
        except Exception as e:
            print('出现错误 --> 错误原因是：' + str(e))

    def outline_func(self):
        try:
            self.outline = OutLine_Window()
            self.outline.show()
        except Exception as e:
            print('出现错误 --> 错误原因是：' + str(e))

    def edge_optimization_func(self):
        try:
            self.edge_Optimiza = Edge_Optimiza_Win()
            self.edge_Optimiza.show()
        except Exception as e:
            print('出现错误 --> 错误原因是：' + str(e))

    def open_img_func(self):
        file_dialog = QFileDialog()
        files_name = file_dialog.getOpenFileNames(self, '选择图片', './',
                                    'PNG(*.png);;JPG(*.jpg);;TIF(*.tif);;ALL(*.*)', 'PNG(*.png)')[0]
        for file_name in files_name:
            if file_name != '':
                # print(type(file_name), file_name)

                self.layers_list.append(file_name.split('/')[-1])  # 记录每一个图层中图像的名称
                self.create_layer(file_name.split('/')[-1])     # 创建一个图层
                self.add_path_to_tree(file_name)  # 调用把路径添加到树控件中
                image = QImage(file_name)
                # image.load(file_name)
                # image = image.scaled(int(self.layer.width()*20.8), int(self.layer.height()*20.8))
                self.layer.setPixmap(QPixmap.fromImage(image))
                self.layer.setAlignment(Qt.AlignCenter)          # 居中显示

                self.current_tree_change()

    def zoom_img_func(self):
        if self.layer != None:
            self.layer.resize(int(self.layer.width()*1.2), int(self.layer.height()*1.2))
            self.layer.move((self.main_Lab.width()-self.layer.width())//2,
                            (self.main_Lab.height()-self.layer.height())//2)

    def lessen_img_func(self):
        if self.layer != None:
            self.layer.resize(int(self.layer.width() * 0.8), int(self.layer.height() * 0.8))
            self.layer.move((self.main_Lab.width() - self.layer.width()) // 2,
                            (self.main_Lab.height() - self.layer.height()) // 2)

    def move_img_func(self):
        if self.layer != None:
            print(self.layer.x())
            print(self.layer.y())
            move_x = self.init_x - self.now_x
            move_y = self.init_y - self.now_y
            self.layer.move(self.layer.x() + move_x, self.layer.y() + move_y)

    def mouse_click(self, p_x, p_y):
        self.main_Lab.my_child = self.layer # 把当前的图层传给main lab，用于获得当前图层的初始相对位置
        if self.layer != None:
            self.init_x = p_x
            self.init_y = p_y

    def down_wheel_func(self):
        if self.layer != None:
            self.layer.move(self.layer.x(), self.layer.y() - self.move_speed)

    def up_wheel_func(self):
        if self.layer != None:
            self.layer.move(self.layer.x(), self.layer.y() + self.move_speed)

    def left_wheel_func(self):
        if self.layer != None:
            self.layer.move(self.layer.x() - self.move_speed, self.layer.y())

    def right_wheel_func(self):
        if self.layer != None:
            self.layer.move(self.layer.x() + self.move_speed, self.layer.y())

    def ctrl_zoom_in_func(self):
        zoom_out_m = 0.95
        if self.layer != None:
            print(self.layer.size())
            self.layer.resize(int(self.layer.width()*zoom_out_m), int(self.layer.height()*zoom_out_m))

    def ctrl_zoom_out_func(self):
        if self.layer != None:
            print(self.layer.size())
            zoom_out_m = 1.05
            if self.layer != None:
                print(self.layer.size())
                self.layer.resize(int(self.layer.width() * zoom_out_m), int(self.layer.height() * zoom_out_m))

    def toPrevious_func(self):  # 测试中
        if self.layer != None:
            print(self.layers_list)
            print(self.tree.currentIndex().data())
            item = self.tree.currentItem()
            print('当前item', item)
            print(item.text(0))
            current_index = self.layers_list.index(item.text(0))
            print(current_index)
            print(self.layers_list[current_index - 1])
            print('ABOVE',self.tree.itemAbove(item).text(0))
            # print('BeLow',self.tree.itemBelow(item).text(0))



    def create_layer(self, name=None):
        # print('create_layer')
        self.layer = QLabel()    # MyLabLayer
        self.layer.setParent(self.main_Lab)
        self.layer.setStyleSheet('background-color:cyan')
        self.layer.setScaledContents(True)
        self.layer.setMouseTracking(True)
        self.layer.resize(500, 500)
        self.layer.move((self.main_Lab.width()-self.layer.width())//2,
                        (self.main_Lab.height()-self.layer.height())//2)
        self.layer.show()
        self.layer.setObjectName(name)
        # self.tree.setCurrentItem()


    def update_status_pos(self, p_x, p_y):
        self.now_x = p_x
        self.now_y = p_y
        self.pos_lab.setText('{}，{}\t'.format(p_x, p_y))
        # now_x,now_y 鼠标相对于main lab的实时位置
        # init_x，init_y 鼠标点击时，相对于main lab的位置
        # main_Lab.my_child_x， main_Lab.my_child_y 图层相对于main lab的最初相对位置
        if self.layer != None and self.main_Lab.isClick and self.main_Lab.isMove:
            move_x = self.now_x - self.init_x
            move_y = self.now_y - self.init_y
            self.layer.move(self.main_Lab.my_child_x + move_x, self.main_Lab.my_child_y + move_y)


    def current_tree_change(self):
        item = self.tree.currentItem()
        if item.text(0) != 'layer':
            for i in self.layers_list:
                self.main_Lab.findChild(QLabel, i).setVisible(False)
            self.main_Lab.findChild(QLabel, item.text(0)).setVisible(True)
            self.layer = self.main_Lab.findChild(QLabel, item.text(0))

    # def toggle_show(self):
    #     item = self.tree.currentItem()
    #     if item.text(0) != 'layer':
    #         print(item.text(0))
    #         print(self.tree.itemClicked(item, 0))
    def move_img_btn_func(self):
        self.setCursor(Qt.ClosedHandCursor)
        self.main_Lab.setCursor(Qt.ClosedHandCursor)
        if self.layer != None:
            self.layer.isMoveBtn = True

    def select_btn_func(self):
        self.setCursor(Qt.ArrowCursor)
        self.main_Lab.setCursor(Qt.ArrowCursor)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        print(QThread.currentThread())
        reply = QMessageBox.question(self, '提示', '确认退出嘛？')
        if reply == QMessageBox.Yes:
            # self.create_ini()
            a0.accept()
            sys.exit(0)
        else:
            a0.ignore()

    def enterEvent(self, a0: QtCore.QEvent) -> None:
        # print('进入主窗体')
        self.grabKeyboard()

    def leaveEvent(self, a0: QtCore.QEvent) -> None:
        # print('离开主窗体')
        self.releaseKeyboard()

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.key() == Qt.Key_F11:
            status = self.windowState()
            if status == Qt.WindowMaximized:
                self.setWindowState(Qt.WindowNoState)
            else:
                self.setWindowState(Qt.WindowMaximized)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = MainWindow()
    w.show()

    sys.exit(app.exec())