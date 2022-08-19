from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1012, 593)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        MainWindow.setMaximumSize(QtCore.QSize(99999, 999999))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        MainWindow.setFont(font)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        MainWindow.setTabletTracking(False)
        MainWindow.setLayoutDirection(QtCore.Qt.LeftToRight)
        MainWindow.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(1012, 568))
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.Widget = QtWidgets.QTabWidget(self.centralwidget)
        self.Widget.setMinimumSize(QtCore.QSize(801, 451))
        font = QtGui.QFont()
        font.setItalic(False)
        self.Widget.setFont(font)
        self.Widget.setFocusPolicy(QtCore.Qt.TabFocus)
        self.Widget.setAcceptDrops(False)
        self.Widget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Widget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.Widget.setIconSize(QtCore.QSize(16, 16))
        self.Widget.setElideMode(QtCore.Qt.ElideLeft)
        self.Widget.setDocumentMode(True)
        self.Widget.setTabsClosable(False)
        self.Widget.setMovable(False)
        self.Widget.setTabBarAutoHide(False)
        self.Widget.setObjectName("Widget")
        self.tab = QtWidgets.QWidget()
        self.tab.setMinimumSize(QtCore.QSize(945, 0))
        self.tab.setObjectName("tab")
        self.gridLayout = QtWidgets.QGridLayout(self.tab)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setMinimumSize(QtCore.QSize(0, 0))
        self.label.setMaximumSize(QtCore.QSize(1007, 517))
        self.label.setStyleSheet("border-image: url(:/picture_background/back_ground.jpg);")
        self.label.setText("")
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.Label = QtWidgets.QLabel(self.tab)
        self.Label.setGeometry(QtCore.QRect(10, 10, 681, 501))
        self.Label.setMinimumSize(QtCore.QSize(680, 500))
        self.Label.setFrameShape(QtWidgets.QFrame.Panel)
        self.Label.setLineWidth(5)
        self.Label.setMidLineWidth(4)
        self.Label.setText("")
        self.Label.setScaledContents(True)
        self.Label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.Label.setObjectName("Label")
        self.Button_stop = QtWidgets.QPushButton(self.tab)
        self.Button_stop.setGeometry(QtCore.QRect(760, 400, 153, 60))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.Button_stop.setFont(font)
        self.Button_stop.setObjectName("Button_stop")
        self.Label_img = QtWidgets.QLabel(self.tab)
        self.Label_img.setEnabled(True)
        self.Label_img.setGeometry(QtCore.QRect(770, 10, 141, 141))
        self.Label_img.setMinimumSize(QtCore.QSize(141, 141))
        self.Label_img.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.Label_img.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.Label_img.setLineWidth(3)
        self.Label_img.setText("")
        self.Label_img.setScaledContents(True)
        self.Label_img.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.Label_img.setObjectName("Label_img")
        self.Label_name = QtWidgets.QLabel(self.tab)
        self.Label_name.setEnabled(True)
        self.Label_name.setGeometry(QtCore.QRect(720, 240, 231, 131))
        self.Label_name.setMinimumSize(QtCore.QSize(231, 131))
        self.Label_name.setFrameShape(QtWidgets.QFrame.Panel)
        self.Label_name.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Label_name.setMidLineWidth(0)
        self.Label_name.setText("")
        self.Label_name.setTextFormat(QtCore.Qt.AutoText)
        self.Label_name.setScaledContents(False)
        self.Label_name.setAlignment(QtCore.Qt.AlignCenter)
        self.Label_name.setObjectName("Label_name")
        self.Label_status = QtWidgets.QLabel(self.tab)
        self.Label_status.setGeometry(QtCore.QRect(720, 170, 231, 61))
        self.Label_status.setMinimumSize(QtCore.QSize(231, 61))
        self.Label_status.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Label_status.setText("")
        self.Label_status.setAlignment(QtCore.Qt.AlignCenter)
        self.Label_status.setObjectName("Label_status")
        self.label.raise_()
        self.Label.raise_()
        self.Button_stop.raise_()
        self.Label_img.raise_()
        self.Label_name.raise_()
        self.Label_status.raise_()
        self.Widget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setGeometry(QtCore.QRect(0, 0, 1021, 581))
        self.label_2.setMinimumSize(QtCore.QSize(1021, 581))
        self.label_2.setStyleSheet("border-image: url(:/picture_background/back_ground.jpg);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.Table = QtWidgets.QTableWidget(self.tab_2)
        self.Table.setGeometry(QtCore.QRect(100, 60, 801, 451))
        self.Table.setMinimumSize(QtCore.QSize(801, 451))
        self.Table.setFrameShape(QtWidgets.QFrame.Box)
        self.Table.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Table.setLineWidth(3)
        self.Table.setObjectName("Table")
        self.Table.setColumnCount(0)
        self.Table.setRowCount(0)
        self.label_13 = QtWidgets.QLabel(self.tab_2)
        self.label_13.setGeometry(QtCore.QRect(0, 0, 1021, 61))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setItalic(False)
        self.label_13.setFont(font)
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.Widget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.label_11 = QtWidgets.QLabel(self.tab_3)
        self.label_11.setGeometry(QtCore.QRect(0, 0, 1021, 581))
        self.label_11.setMinimumSize(QtCore.QSize(1021, 581))
        self.label_11.setStyleSheet("border-image: url(:/picture_background/back_ground.jpg);")
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.label_14 = QtWidgets.QLabel(self.tab_3)
        self.label_14.setGeometry(QtCore.QRect(0, 0, 1021, 61))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setItalic(False)
        self.label_14.setFont(font)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.Table_1 = QtWidgets.QTableWidget(self.tab_3)
        self.Table_1.setGeometry(QtCore.QRect(100, 60, 831, 451))
        self.Table_1.setMinimumSize(QtCore.QSize(801, 451))
        self.Table_1.setFrameShape(QtWidgets.QFrame.Box)
        self.Table_1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Table_1.setLineWidth(3)
        self.Table_1.setObjectName("Table_1")
        self.Table_1.setColumnCount(0)
        self.Table_1.setRowCount(0)
        self.Widget.addTab(self.tab_3, "")
        self.gridLayout_2.addWidget(self.Widget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.Widget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Final Project- Nguyen Duc Hien 18161219"))
        self.Button_stop.setText(_translate("MainWindow", "Stop Program"))
        self.Widget.setTabText(self.Widget.indexOf(self.tab), _translate("MainWindow", "Register"))
        self.label_13.setText(_translate("MainWindow", "List Of Employee Register"))
        self.Widget.setTabText(self.Widget.indexOf(self.tab_2), _translate("MainWindow", "Register List "))
        self.label_14.setText(_translate("MainWindow", "Report Of Statistics Employee  For The Month"))
        self.Widget.setTabText(self.Widget.indexOf(self.tab_3), _translate("MainWindow", "Statistic Report"))
import back_ground_rc