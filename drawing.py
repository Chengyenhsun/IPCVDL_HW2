from PyQt5 import QtCore, QtGui, QtWidgets


class DrawingScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent=None):
        super(DrawingScene, self).__init__(parent)
        self.pos_xy = []

    def mousePressEvent(self, event):
        pos_tmp = event.scenePos()
        self.pos_xy.append(pos_tmp)
        self.update()

    def mouseMoveEvent(self, event):
        pos_tmp = event.scenePos()
        self.pos_xy.append(pos_tmp)
        self.update()

    def mouseReleaseEvent(self, event):
        pos_test = QtCore.QPointF(-1, -1)
        self.pos_xy.append(pos_test)
        self.update()

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0), 2, QtCore.Qt.SolidLine)
        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == QtCore.QPointF(-1, -1):
                    point_start = QtCore.QPointF(-1, -1)
                    continue
                if point_start == QtCore.QPointF(-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start, point_end)
                point_start = point_end


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(130, 90, 391, 321))
        self.graphicsView.setObjectName("graphicsView")

        self.scene = DrawingScene()
        self.graphicsView.setScene(self.scene)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Handwriting Board"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
