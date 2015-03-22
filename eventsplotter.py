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
        self.linkedFileName = None
        self.specialFileName = None
        self.eventsData = None
        self.specialEventsData = None
        self.plottedEvent = None
        self.eventCommentaries = defaultdict(lambda: "")

        # GUI initialization
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
        self.mainLayout.setRowMinimumHeight(0, 50)
        self.mainLayout.addLayout(self.plotInfo, 0, 0, 1, 3)
        self.mainLayout.addWidget(self.plotWindow, 1, 0, 1, 3)
        self.mainLayout.addWidget(self.btnPrevEvent, 2, 1)
        self.mainLayout.addWidget(self.btnNextEvent, 2, 2)
        self.mainLayout.addWidget(self.txtComment, 2, 0)

        centralWidget.setLayout(self.mainLayout)
        self.setCentralWidget(centralWidget)
        self.setWindowTitle("Events plotter")

    def createMenuBar(self):
        """ Adds menu bar to main window """
        actionSelectEventsFile = QAction("&Select CSV-file...", self)
        # actionSelectSpecialEventsFile = QAction("&Select all day events...", self)

        optionsMenu = self.menuBar().addMenu("&Options")
        optionsMenu.addAction(actionSelectEventsFile)
        # optionsMenu.addAction(actionSelectSpecialEventsFile)

        actionSelectEventsFile.triggered.connect(lambda: self.showChooseLinkedFileDialog())
        # actionSelectSpecialEventsFile.triggered.connect(lambda: self.showChooseAllDayFileDialog())

    def bindEvents(self):
        self.btnPrevEvent.clicked.connect(self.plotPrevEvent)
        self.btnNextEvent.clicked.connect(self.plotNextEvent)

    def plotEvent(self, ahead=True):
        """ Increments or decrements current event index and plots new graph onto canvas.
        """
        leavedEvent = self.getSameTimeEvents()
        # step = len(leavedEvent)
        joinedNames = self.CompoundEventSplitter.join(leavedEvent)

        if ahead:
            self.plottedEvent = min(self.plottedEvent + 1, len(self.eventNames) - 1)
        else:
            self.plottedEvent = max(0, self.plottedEvent - 1)

        comment = str(self.txtComment.text())
        if comment:
            self.eventCommentaries[joinedNames] = comment

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

    def prepareDialog(self):
        """ Helper function for file choosing menu actions"""
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("CSV-files (*.csv)")
        return dlg

    def showChooseLinkedFileDialog(self):
        """ Opens a file choosing dialog to select CSV-file with linked events info """
        dlg = self.prepareDialog()
        if dlg.exec():
            self.linkedFileName = dlg.selectedFiles()[0]
            self.eventsData = self.readCsv(self.linkedFileName, delimiter=',')
            self.eventNames = self.eventsData.Event.unique()
            self.plottedEvent = 0
            self.plot()
            self.fillPlotInfo()

    def showChooseAllDayFileDialog(self):
        dlg = self.prepareDialog()
        if dlg.exec():
            self.specialFileName = dlg.selectedFiles()[0]
            self.specialEventsData = self.readCsv(self.specialFileName, delimiter=',')

    def readCsv(self, filename, delimiter):
        if filename is None:
            return

        df = pd.read_csv(filename, delimiter=delimiter)
        return self.resolve_duplicates(df)

    def resolve_duplicates(self, df):
        """ Adds data postfixes to duplicated events names """
        event_names = df.Event.unique()
        processed = list()

        for name in event_names:
            dfslice = df[df.Event == name]
            keys, groups = list(), list()

            for k, g in dfslice.groupby(['DateUTC', 'TimeUTC']):
                keys.append(k)
                groups.append(g)

            # if len(keys) > 1:
            for (date, time), g in zip(keys, groups):
                procdf = g
                procdf["Event"] += " ({} {})".format(date, time)
                processed.append(procdf)

            # else:
            #     processed.extend(groups)

        newdf = pd.concat(processed)
        return newdf

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
        dates, times = current_event.DateUTC.unique(), current_event.TimeUTC.unique()
        date, time = dates[0], times[0]

        same_time_events = df[(df.DateUTC == date) & (df.TimeUTC == time)].Event
        return list(same_time_events.unique())

    def fillPlotInfo(self, ignore=None):
        """
        """
        if ignore is None:
            ignore = ['Timestamp', 'DateAndTime', 'Date', 'Time', 'Event']

        df = self.eventsData

        event_infos = dict()
        for name in self.getSameTimeEvents():
            current_event = df[df.Event == name]
            date = current_event.Date.iloc[0]
            event_infos[name] = [(k, v) for k, v in current_event.iloc[0].iteritems() if k not in ignore]

        # remove widgets with info from previous event
        while True:
            widget_item = self.plotInfo.takeAt(0)
            if widget_item is None:
                break
            widget_item.widget().hide()
            self.plotInfo.removeWidget(widget_item.widget())
            widget_item.widget().deleteLater()

        last_row, info_length, info_labels = None, None, None

        # dynamically fills info layout with widgets
        for row, name in enumerate(event_infos):
            labelEventName = QLabel(name)
            labelEventName.setFont(QFont("Palatino", 10))
            self.plotInfo.addWidget(labelEventName, row + 1, 0)
            info = event_infos[name]

            for col, (label, value) in enumerate(info, 1):
                if row == 0:
                    # first event processing so need add column headers
                    self.plotInfo.addWidget(QLabel(label + ":"), 0, col)
                self.plotInfo.addWidget(QLabel(str(value)), row + 1, col)

            last_row = row
            info_labels = info
            info_length = len(info)

        # TODO: finish with All Day events
        # ev = self.specialEventsData
        # if isinstance(ev, pd.DataFrame):
        #     ev = ev[ev.Date == date]
        #
        #     if not ev.empty:
        #         last_row += 1
        #         self.plotInfo.addWidget(QLabel("Special events"), last_row, 0, info_length, 1)
        #
        #         labelEventName = QLabel(name)
        #         labelEventName.setFont(QFont("Palatino", 10))
        #         self.plotInfo.addWidget(labelEventName, last_row + 1, 0)
        #
        #         for col, (label, value) in enumerate(info, 1):
        #             self.plotInfo.addWidget(QLabel(str(value)), row + 1, col)

        self.plotInfo.setRowStretch(last_row + 1, 10)
        self.repaint()

    def closeEvent(self, *args, **kwargs):
        """ Form closing event handler.

            Saves all commentraies that have been entered into CSV-file.
        """
        if not self.linkedFileName:
            return

        if not any(self.eventCommentaries.values()):
            return

        with open(self.linkedFileName.replace(".csv", "") + "_comments.csv", "w") as f:
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
        dates = fed.DateAndTime.values

        # plot graph and setup plotting surface parametes
        ax.hold(True)
        ax = fed.plot(x=xcol, y=ycol, ax=ax, rot=20)
        ax.tick_params(axis='x', which='major', labelsize=9)
        ax.tick_params(axis='y', which='major', labelsize=10)
        yformatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        ax.yaxis.set_major_formatter(yformatter)

        # refresh canvas
        self.canvas.draw()
        self.figure.tight_layout(pad=0.5)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ep = EventsPlotter((800, 600))
    ep.show()
    app.exec_()