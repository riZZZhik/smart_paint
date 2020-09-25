# System imports
import sys
from glob import glob

# Interface imports
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtWidgets import QApplication, QMainWindow

# Import arrays module
import numpy as np

# Module imports
from .photo.utils import is_image


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


class Interface(QMainWindow, Design):  # TODO: Class description
    """Create an interface"""
    def __init__(self):
        """Initialize interface parameters"""
        self.app = QApplication([])
        super().__init__()

        self.setup_ui()
        self.setAcceptDrops(True)
        self.setWindowTitle('Smart Paint')

        self.styleID, self.styles = None, []
        self.path, self.result = None, None

        self.fill_styles_images()
        self.show()

        # TODO: ML module

    def stylize(self):
        """Function to stylize user image using machine learning prediction"""
        if self.styleID and self.path:
            self.result = np.zeros((256, 256, 3))  # TODO: ML predict

            shape = self.result.shape
            self.rightItem.setPixmap(QPixmap(QImage(self.result,
                                                    shape[1], shape[0], shape[1] * 3, QImage.Format_RGB888)))

    @staticmethod
    def drag_enter_event(event):
        """Сhecking if there are elements in the drag"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def drop_event(self, event):
        """Show the dropped image and its stylization
        :param event: Event module from PyQT5 library"""
        for _, path in enumerate(event.mimeData().urls()):
            path = path.toString()[7:]
            if is_image(path):
                self.path = path
                self.leftItem.setPixmap(QPixmap(self.path))
                self.show()
            else:
                print("It is not image, try again")  # TODO: Display it

        self.stylize()

    def style_click(self, style_id):
        """Show stylized image with the clicked style
        :param style_id: Id of selected style"""
        self.styleID = style_id
        for style in self.styles:
            style.setStyleSheet("QLabel {border-width: 0;}")
        self.styles[style_id].setStyleSheet("QLabel {"
                                            "border-style: solid;"
                                            "border-width: 3px;"
                                            "border-color: gray;"
                                            "}")
        self.stylize()


if __name__ is "__main__":
    interface = Interface()
    sys.exit(interface.app.exec())
