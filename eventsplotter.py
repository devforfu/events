import os
import sys
import csv
from datetime import datetime
from collections import defaultdict

import PyQt4
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import pandas as pd


class EventsPlotter(QMainWindow):

    CompoundEventSplitter = " :: "

    def __init__(self, size):
        super(EventsPlotter, self).__init__()
        self.fileName = None
        self.eventsData = None
        self.plottedEvent = None
        self.eventCommentaries = defaultdict(lambda: "")
        self.createLayout()
        self.bindEvents()
        self.setMinimumSize(*size)

    def createLayout(self):
        """ Fills main window with widges and layouts """
        centralWidget = QWidget()
        self.mainLayout = QGridLayout()
        self.plotWindow = PlotWindow(parent=self)
        self.plotInfo = QGridLayout()
        self.txtComment = QLineEdit()
        self.btnPrevEvent = QPushButton("<- Prev event")
        self.btnNextEvent = QPushButton("Next event ->")

        self.txtComment.setPlaceholderText("Enter plot comment here...")
        self.btnPrevEvent.setMinimumWidth(100)
        self.btnNextEvent.setMinimumWidth(100)

        self.createMenuBar()

        self.mainLayout.setRowStretch(0, 10)
        self.mainLayout.setColumnStretch(0, 10)
        self.mainLayout.addWidget(self.plotWindow, 0, 0, 1, 3)
        self.mainLayout.addWidget(self.btnPrevEvent, 1, 1)
        self.mainLayout.addWidget(self.btnNextEvent, 1, 2)
        self.mainLayout.addWidget(self.txtComment, 1, 0)
        # self.mainLayout.addLayout(self.plotInfo, 0, 1, 1, 2)
        # self.mainLayout.addLayout(self.plotInfo, 0, 1, 1, 2)

        centralWidget.setLayout(self.mainLayout)
        self.setCentralWidget(centralWidget)
        self.setWindowTitle("Events plotter")

    def createMenuBar(self):
        """ Adds menu bar to main window """
        actionSelectFile = QAction("&Select CSV-file...", self)
        optionsMenu = self.menuBar().addMenu("&Options")
        optionsMenu.addAction(actionSelectFile)
        actionSelectFile.triggered.connect(lambda: self.showChooseFileDialog())

    def bindEvents(self):
        self.btnPrevEvent.clicked.connect(self.plotPrevEvent)
        self.btnNextEvent.clicked.connect(self.plotNextEvent)

    def plotEvent(self, ahead=True):
        """ Increments or decrements current event index and plots new graph onto canvas.
        """
        leavedEvent = self.getSameTimeEvents()
        leavedEvent = self.CompoundEventSplitter.join(leavedEvent)

        if ahead:
            self.plottedEvent = min(self.plottedEvent + 1, len(self.eventNames) - 1)
        else:
            self.plottedEvent = max(0, self.plottedEvent - 1)

        comment = self.txtComment.text()
        if comment:
            self.eventCommentaries[leavedEvent] = comment

        enteredEvent = self.getSameTimeEvents()
        enteredEvent = self.CompoundEventSplitter.join(enteredEvent)

        comment = self.eventCommentaries[enteredEvent]
        self.txtComment.setText(comment)
        self.plot()
        self.fillPlotInfo()

    def plotNextEvent(self):
        """ Moves to next plotted event """
        self.plotEvent()

    def plotPrevEvent(self):
        """ Moves to previous plotted event """
        self.plotEvent(ahead=False)

    def showChooseFileDialog(self):
        """ Opens a file choosing dialog to select CSV-file with linked events info """
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("CSV-files (*.csv)")
        if dlg.exec():
            self.fileName = dlg.selectedFiles()[0]
            self.readCsv(delimiter=',')
            self.plottedEvent = 0
            self.eventNames = self.eventsData.Event.unique()
            self.plot()
            self.fillPlotInfo()

    def readCsv(self, delimiter):
        if self.fileName is None:
            return

        # df = pd.read_csv(self.fileName, delimiter=delimiter)
        # fmt = "%Y%m%d %H:%M:%S:%f"
        # df["DateAndTime"] = [pd.to_datetime(ts, format=fmt) for ts in df.Timestamp]
        self.eventsData = pd.read_csv(self.fileName, delimiter=delimiter)

    def plot(self):
        """ Plots event graphs if events data was loaded.

            Delegates most of functionality to PlotWindow class
        """
        if self.eventsData is None or self.eventNames is None:
            return
        current_events = self.getSameTimeEvents()
        self.plotWindow.clearPlot()
        self.plotWindow.plot(self.eventsData, event_name=current_events,
                             splitter=self.CompoundEventSplitter,
                             xcol="DateAndTime", ycol=["Ask price", "Bid price"])

    def getSameTimeEvents(self):
        """ Gets events that occured at same time with curretly plotted event """
        df = self.eventsData
        event_name = self.eventNames[self.plottedEvent]
        current_event = df[df.Event == event_name]
        dates, times = current_event.Date.unique(), current_event.Time.unique()
        date, time = dates[0], times[0]

        same_time_events = df[(df.Date == date) & (df.Time == time)].Event
        return list(same_time_events.unique())

    def fillPlotInfo(self, ignore=None):
        if ignore is None:
            ignore = ['Timestamp', 'DateAndTime', 'Event']

        df = self.eventsData
        current_event = df[df.Event == self.eventNames[self.plottedEvent]]
        infos = [(k, v) for k, v in current_event.iloc[0].iteritems() if k not in ignore]

        while True:
            widget_item = self.plotInfo.takeAt(0)
            if widget_item is None:
                break
            self.plotInfo.removeWidget(widget_item.widget())
            widget_item.widget().deleteLater()

        last = None
        for row, (label, value) in enumerate(infos):
            self.plotInfo.addWidget(QLabel(label + ":"), row, 0)
            self.plotInfo.addWidget(QLabel(str(value)), row, 1)
            last = row

        self.plotInfo.setRowStretch(last + 1, 10)

    def closeEvent(self, *args, **kwargs):
        """ Form closing event handler.

            Saves all commentraies that have been entered into CSV-file.
        """
        if not self.fileName:
            return

        with open(self.fileName.replace(".csv", "") + "_comments.csv", "w") as f:
            w = csv.writer(f)
            w.writerow(['Event', 'Comment'])
            for event in self.eventCommentaries:
                if not self.eventCommentaries[event]:
                    continue
                w.writerow([event, self.eventCommentaries[event]])


class PlotWindow(QDialog):
    """ Intregates matplotlib canvas into Qt window """

    def __init__(self, width=14, height=10, parent=None):
        super(PlotWindow, self).__init__(parent)

        self.figure = plt.figure(figsize=(width, height))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def clearPlot(self):
        plt.cla()

    def plot(self, df, event_name, xcol, ycol, splitter=" / "):
        """ Plots graph for specified event """
        concat_names = ""
        if isinstance(event_name, list):
            # handle case with several events occurred in same time
            concat_names = splitter.join([name for name in event_name])
            event_name = event_name[0]

        ax = self.figure.add_subplot(1, 1, 1)
        fed = df[df.Event == event_name]
        fed = pd.DataFrame({"Ask price": fed["Ask price"].values,
                            "Bid price": fed["Bid price"].values},
                           #columns=["Ask price", "Bid price"],
                           index=fed.DateAndTime)

        fed = df[df.Event == event_name]
        dates = fed.DateAndTime.values
        # first, last = dates[0], dates[-1]

        # plot graph and setup plotting surface parametes
        ax.hold(True)
        ax = fed.plot(x=xcol, y=ycol, ax=ax, rot=20)
        # data_range = pd.date_range(first, last, freq='30S').astype(datetime)
        # ax.xaxis.set_ticks(data_range)
        # ax.xaxis.set_ticklabels(data_range)
        ax.tick_params(axis='x', which='major', labelsize=9)
        ax.tick_params(axis='y', which='major', labelsize=10)
        yformatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        ax.yaxis.set_major_formatter(yformatter)
        ax.set_title(concat_names if concat_names else event_name)

        # refresh canvas
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ep = EventsPlotter((800, 600))
    ep.show()
    app.exec_()