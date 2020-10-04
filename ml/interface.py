# System imports
import sys
from glob import glob

# Arrays imports
import numpy as np
from .photo.utils import is_image, save_img

# Interface imports
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtWidgets import QApplication, QMainWindow


# noinspection PyTypeChecker
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
        """Setup main user interface"""
        # Init main window parameters
        self.setObjectName("Form")
        self.resize(770, 310)
        self.setAcceptDrops(True)

        # Create download button
        self.download = QtWidgets.QPushButton(self)
        self.download.setGeometry(QtCore.QRect(510, 270, 250, 30))
        self.download.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.download.setObjectName("download")

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

        # Set widgets' texts
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Form", "Smart Paint"))
        self.download.setText(_translate("Form", "PushButton"))
        self.download.setText(_translate("Form", 'Download image'))
        QtCore.QMetaObject.connectSlotsByName(self)


class Interface(QMainWindow, Design):  # TODO: Class description
    """Create an interface"""

    def __init__(self):
        """Initialize interface parameters"""
        self.app = QApplication([])
        super().__init__()

        self.style_id, self.styles = 0, []
        self.path, self.result = None, None

        self.setup_ui()
        self.fill_styles_images()

        self.download.clicked.connect(self.save_image)

        self.show()

        # TODO: ML module

    def stylize(self):
        """Function to stylize user image using machine learning prediction"""
        if self.path:
            self.result = np.random.randint(255, size=(256, 256, 3), dtype=np.uint8)  # TODO: ML predict

            shape = self.result.shape
            self.rightItem.setPixmap(QPixmap(QImage(self.result,
                                                    shape[1], shape[0], shape[1] * 3, QImage.Format_RGB888)))

    @staticmethod
    def dragEnterEvent(event, **kwargs):
        """Сhecking if there are elements in the drag

        Args:
            event (PyQt5.QtGui.QDragEnterEvent): Event module from PyQT5 library
        """
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Show the dropped image and its stylization

        Args:
            event (PyQt5.QtGui.QDropEvent):  Event module from PyQT5 library
        """
        for path in event.mimeData().urls():
            path = path.toString()[7:]
            if is_image(path):
                self.path = path
                self.leftItem.setPixmap(QPixmap(self.path))
                self.show()
            else:
                print("It is not image, try again")  # TODO: Display this

        self.stylize()

    def save_image(self, _):
        """Save stylized image to disk"""
        out_path = "".join(self.path.split(".")[:-1])
        out_path += "_result."
        out_path += self.path.split(".")[-1]

        save_img(out_path, self.result)

    def style_click(self, style_id: int):
        """Show stylized image with the clicked style

        Args:
            style_id (int): Id of selected style:
        """
        if style_id != self.style_id:
            self.style_id = style_id
            for style in self.styles:
                style.setStyleSheet("QLabel {border-width: 0;}")
            self.styles[style_id].setStyleSheet("QLabel {"
                                                "border-style: solid;"
                                                "border-width: 3px;"
                                                "border-color: gray;"
                                                "}")
            self.stylize()

    def fill_styles_images(self, styles_dir: str = 'styles/'):
        """ Fill style images to middle layout

        Args:
            styles_dir (str): Directory with style images
        """
        images_paths = sorted(glob(styles_dir + "*.jpg") + glob(styles_dir + "*.png"))
        images_num = len(images_paths)
        for path, pos, style_click in zip(images_paths,
                                          self._get_styles_positions(images_num), self._get_styles_funcs(images_num)):
            style = QtWidgets.QLabel(self.layoutWidget)
            style.mousePressEvent = style_click

            style.setText("")
            style.setScaledContents(True)
            style.setPixmap(QPixmap(path))

            self.styles.append(style)
            self.midLayout.addWidget(style, pos[0], pos[1])

        self.style_click(0)

    def _get_styles_funcs(self, num: int):
        """Generate click functions for each style

        Args:
            num (int): Number of style images
        Returns:
            list of style functions
        """
        # style_click = self.style_click
        # for i in range(num):
        #     def click(_):
        #         style_click(i)
        #     yield click  # TODO: Fix this

        style_click = self.style_click

        def click_0(_):
            style_click(0)

        def click_1(_):
            style_click(1)

        def click_2(_):
            style_click(2)

        def click_3(_):
            style_click(3)

        def click_4(_):
            style_click(4)

        funcs = [click_0, click_1, click_2, click_3, click_4]
        return funcs

    @staticmethod
    def _get_styles_positions(num):
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]  # TODO: Optimize this
        return positions[:num]


if __name__ is "__main__":
    interface = Interface()
    sys.exit(interface.app.exec())
