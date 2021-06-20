# -*- coding: utf-8 -*-
from functools import partial

import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWidgets
import seaborn as sn
import pandas as pd
from PyQt5.QtWidgets import QTableWidgetItem

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import cohen_kappa_score

iris = pd.read_csv('IRIS.csv')
X = iris.drop('species', axis=1)
y = iris['species']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
pd.DataFrame({'Actual': y_val, 'Predicted': y_pred})

acc = accuracy_score(y_val, y_pred, normalize=True)
acc_string = '{0:0.8f}'.format(acc)

f1mac = f1_score(y_val, y_pred, average='macro')
f1mac_string = '{0:0.8f}'.format(f1mac)

f1mic = f1_score(y_val, y_pred, average='micro')
f1mic_string = '{0:0.8f}'.format(f1mic)

f1wei = f1_score(y_val, y_pred, average='weighted')
f1wei_string = '{0:0.8f}'.format(f1wei)

kappa = cohen_kappa_score(y_val, y_pred)
kappa_string = '{0:0.8f}'.format(kappa)

data = {'y_Actual': y_val, 'y_Predicted': y_pred}
df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
confMat = sn.heatmap(confusion_matrix, annot=True)
plt.show()

w, h = 4, 1
newInput = [[0 for x in range(w)] for y in range(h)]


class Ui_MainWindow(object):

    def prediction(self):

        oznitelik1 = float(self.lineEdit.text())
        oznitelik2 = float(self.lineEdit_2.text())
        oznitelik3 = float(self.lineEdit_3.text())
        oznitelik4 = float(self.lineEdit_4.text())

        newInput[0][0] = oznitelik1
        newInput[0][1] = oznitelik2
        newInput[0][2] = oznitelik3
        newInput[0][3] = oznitelik4

        predicted = model.predict(newInput)
        if predicted == 0:
            self.label_13.setText("0 (Iris-setosa)")
        elif predicted == 1:
            self.label_13.setText("1 (Iris-versicolor)")
        elif predicted == 2:
            self.label_13.setText("2 (Iris-virginica)")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1122, 891)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.button = QtWidgets.QPushButton(self.centralwidget)
        self.button.setGeometry(QtCore.QRect(510, 640, 93, 28))
        self.button.setObjectName("button")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(940, 180, 91, 16))
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setObjectName("label")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(20, 20, 520, 461))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setRowCount(150)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(860, 180, 61, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(810, 200, 121, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(940, 200, 91, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(810, 220, 111, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(810, 240, 121, 16))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(880, 260, 55, 16))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(940, 220, 91, 16))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(940, 240, 91, 16))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(940, 260, 91, 16))
        self.label_10.setObjectName("label_10")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(190, 600, 171, 22))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(380, 600, 171, 22))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(570, 600, 171, 22))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(760, 600, 171, 22))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(470, 700, 61, 16))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(540, 700, 101, 16))
        self.label_13.setText("")
        self.label_13.setObjectName("label_13")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.alldata = iris
        numRows = 150
        self.tableWidget.setColumnCount(len(self.alldata.columns))
        self.tableWidget.setRowCount(numRows)
        self.tableWidget.setHorizontalHeaderLabels(self.alldata.columns)

        for i in range(numRows):
            for j in range(len(self.alldata.columns)):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.alldata.iat[i, j])))

        self.tableWidget.resizeColumnsToContents()

        self.button.clicked.connect(partial(self.prediction))

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button.setText(_translate("MainWindow", "Uygula"))
        self.label.setText(_translate("MainWindow", acc_string))
        self.label_2.setText(_translate("MainWindow", "Accuracy :"))
        self.label_3.setText(_translate("MainWindow", "F-Measure(Macro) :"))
        self.label_4.setText(_translate("MainWindow", f1mac_string))
        self.label_5.setText(_translate("MainWindow", "F-Measure(Micro) :"))
        self.label_6.setText(_translate("MainWindow", "F-Measure(Weight):"))
        self.label_7.setText(_translate("MainWindow", "Kappa:"))
        self.label_8.setText(_translate("MainWindow", f1mic_string))
        self.label_9.setText(_translate("MainWindow", f1wei_string))
        self.label_10.setText(_translate("MainWindow", kappa_string))
        self.label_12.setText(_translate("MainWindow", "Tahmin ="))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
