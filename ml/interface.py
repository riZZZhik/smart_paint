# Interface imports
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QCursor


class Design(object):
    def __init__(self):
        """Initialize main window design"""
        self.download = None

        self.layoutWidget = None
        self.horizontalLayout = None

        self.leftItem = None
        self.midLayout = None
        self.rightItem = None

    def setup_ui(self):
        """Setup main user interface
        :param form: QMainWindow class
        """
        # Init main window parameters
        self.setObjectName("Form")
        self.resize(770, 310)

        # Create download button
        self.download = QtWidgets.QPushButton(self)
        self.download.setGeometry(QtCore.QRect(510, 270, 250, 30))
        self.download.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.download.setObjectName("download")
        self.download.setText('Download image')

        # Create main QWidget object
        self.layoutWidget = QtWidgets.QWidget(self)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 750, 250))
        self.layoutWidget.setObjectName("layoutWidget")

        # Create main interface layout
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        # Create left QWidget object with base image
        self.leftItem = QtWidgets.QLabel(self.layoutWidget)
        self.leftItem.setEnabled(True)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.leftItem.sizePolicy().hasHeightForWidth())
        self.leftItem.setSizePolicy(size_policy)
        self.leftItem.setMinimumSize(QtCore.QSize(250, 250))
        self.leftItem.setMaximumSize(QtCore.QSize(250, 250))
        self.leftItem.setText("")
        self.leftItem.setScaledContents(True)
        self.leftItem.setObjectName("leftItem")
        self.horizontalLayout.addWidget(self.leftItem)

        # Create middle QWidget object with style picker
        self.midLayout = QtWidgets.QGridLayout()
        self.midLayout.setObjectName("midLayout")
        self.horizontalLayout.addLayout(self.midLayout)

        # Create right QWidget object with result image
        self.rightItem = QtWidgets.QLabel(self.layoutWidget)
        self.rightItem.setMinimumSize(QtCore.QSize(250, 250))
        self.rightItem.setMaximumSize(QtCore.QSize(250, 250))
        self.rightItem.setText("")
        self.rightItem.setScaledContents(True)
        self.rightItem.setObjectName("rightItem")
        self.horizontalLayout.addWidget(self.rightItem)

        # Customize settings
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Form", "Form"))
        self.download.setText(_translate("Form", "PushButton"))
        QtCore.QMetaObject.connectSlotsByName(self)