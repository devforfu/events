import os
import sys
import csv
import math
from datetime import datetime
from collections import defaultdict

import PyQt4
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def stable_unique(items):
    last = items[0]
    uniq = [last]
    for item in items[1:]:
        if item == last:
            continue
        uniq.append(item)
        last = item
    return uniq


class EventsPlotter(QMainWindow):

    CompoundEventSplitter = " :: "

    IgnoredColumns = ('Timestamp', 'DateAndTime', 'Date', 'Time', 'Event',
                      'Ask price', 'Bid price', 'Ask volume', 'Bid volume',
                      'BreakoutStartDate', 'BreakoutEndDate', 'BreakoutStartPrice',
                      'BreakoutEndPrice', 'BreakoutPriceDelta')

    def __init__(self, size):
        super(EventsPlotter, self).__init__()
        self.linkedFileName = None
        self.specialFileName = None
        self.eventsData = None
        self.specialEventsData = None
        self.plottedEvent = None
        self.eventCommentaries = defaultdict(lambda: "")
        self.sameTimeEvents = None

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
        actionSelectEventsFile = QAction("Select &timed events...", self)
        actionSelectSpecialEventsFile = QAction("Select &special events...", self)

        optionsMenu = self.menuBar().addMenu("&Options")
        optionsMenu.addAction(actionSelectEventsFile)
        optionsMenu.addAction(actionSelectSpecialEventsFile)

        actionSelectEventsFile.triggered.connect(lambda: self.showChooseLinkedFileDialog())
        actionSelectSpecialEventsFile.triggered.connect(lambda: self.showChooseAllDayFileDialog())

    def bindEvents(self):
        self.btnPrevEvent.clicked.connect(self.plotPrevEvent)
        self.btnNextEvent.clicked.connect(self.plotNextEvent)

    def plotEvent(self, ahead=True):
        """ Increments or decrements current event index and plots new graph onto canvas.
        """
        leavedEvents = self.sameTimeEvents[self.plottedEvent]
        joinedNames = self.CompoundEventSplitter.join(leavedEvents)
        self.plottedEvent = {
            True: min(self.plottedEvent + 1, len(self.sameTimeEvents) - 1),
            False: max(0, self.plottedEvent - 1)
        }[ahead]

        comment = str(self.txtComment.text())
        if comment:
            self.eventCommentaries[leavedEvents] = comment

        enteredEvent = self.sameTimeEvents[self.plottedEvent]
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
            self.eventsData = self.resolve_duplicates(self.eventsData)
            self.sameTimeEvents = self.groupEvents()
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
        df["DateAndTime"] = pd.to_datetime(df.DateAndTime)
        return df

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

            for (date, time), g in zip(keys, groups):
                procdf = g
                procdf["Event"] += " ({} {})".format(date, time)
                processed.append(procdf)

        newdf = pd.concat(processed).sort(['Date', 'Time'])

        return newdf

    def plot(self):
        """ Plots event graphs if events data was loaded.

            Delegates most of functionality to PlotWindow class
        """
        if self.eventsData is None or self.sameTimeEvents is None:
            return

        currentEvents = self.sameTimeEvents[self.plottedEvent]
        self.plotWindow.clearPlot()
        self.plotWindow.plot(self.eventsData, event_name=currentEvents,
                             splitter=self.CompoundEventSplitter,
                             xcol="DateAndTime", ycol=["Ask price", "Bid price"])

    def groupEvents(self):
        if not isinstance(self.eventsData, pd.DataFrame):
            raise TypeError("Cannot group events by dates")

        df = self.eventsData
        allSameTimeEvents = list()

        for eventName in stable_unique(df.Event.values):
            currentEvent = df[df.Event == eventName]
            dates, times = currentEvent.DateUTC.unique(), currentEvent.TimeUTC.unique()
            date, time = dates[0], times[0]

            same_time_events = df[(df.DateUTC == date) & (df.TimeUTC == time)].Event
            uniq = tuple(same_time_events.unique())
            if uniq not in allSameTimeEvents:
                allSameTimeEvents.append(uniq)

        return allSameTimeEvents

    def fillPlotInfo(self, ignore=None):
        """
        """
        # if ignore is None:
        #     ignore = EventsPlotter.IgnoredColumns

        df = self.eventsData
        eventInfos = dict()
        date = None

        columns_order = ['Currency', 'Importance', 'ActualForecastDiff', 'Actual',
                         'Forecast', 'PreviousForecastDiff', 'Previous', 'DateUTC', 'TimeUTC']

        for name in self.sameTimeEvents[self.plottedEvent]:
            current_event = df[df.Event == name]
            current_event = current_event[columns_order]
            if date is None:
                date = current_event.DateUTC.iloc[0]
            # eventInfos[name] = [(k, v) for k, v in current_event.iloc[0].iteritems() if k not in ignore]
            eventInfos[name] = [(k, v) for k, v in current_event.iloc[0].iteritems()]

        # special (several days) events
        se = self.specialEventsData
        if isinstance(se, pd.DataFrame) and not se.empty:
            se = se[se.Date == date.replace('-', '.')]
            for name in se.Event:
                current_event = se[se.Event == name]
                name += " (" + current_event.Time.iloc[0] + ")"
                eventInfos[name] = [(k, v) for k, v in current_event.iloc[0].iteritems() if k not in ignore]
                eventInfos[name].append(("DateUTC", "-"))
                eventInfos[name].append(("TimeUTC", "-"))

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
        for row, name in enumerate(eventInfos):
            labelEventName = QLabel(name)
            labelEventName.setFont(QFont("Palatino", 10))
            self.plotInfo.addWidget(labelEventName, row + 1, 0)
            info = eventInfos[name]

            nextStyle = "QLabel {{ background-color: {}; color: black; }}"
            nextColor = "transparent"
            for col, (label, value) in enumerate(info, 1):
                if row == 0:
                    # first event processing so need add column headers
                    self.plotInfo.addWidget(QLabel(label + ":"), 0, col)

                fmt = nextStyle.format(nextColor)
                widget = QLabel(str(value))
                widget.setStyleSheet(fmt)

                if label in ("ActualForecastDiff", "PreviousForecastDiff"):
                    nextColor = {
                        '>': '#32cd32', '=': '#ffffff', '<': '#ed4337'
                    }[value]

                else:
                    nextColor = 'transparent'

                self.plotInfo.addWidget(widget, row + 1, col)

            last_row = row
            info_labels = info
            info_length = len(info)

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
        if isinstance(event_name, (tuple, list)):
            # handle case with several events occurred in same time
            concat_names = splitter.join([name for name in event_name])
            event_name = event_name[0]

        ax = self.figure.add_subplot(1, 1, 1)

        fed = df[df.Event == event_name]
        ymin = math.ceil(fed[ycol].min().min() * 1000) / 1000.0 - 0.0005
        ymax = math.ceil(fed[ycol].max().max() * 1000) / 1000.0 + 0.0005
        dates = fed.DateAndTime.values

        # plot graph and setup plotting surface parametes
        ax.hold(True)
        ax = fed.plot(x=xcol, y=ycol, ax=ax, rot=20)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d/%m/%Y %H:%M:%S"))

        # breakouts plotting
        self.draw_breakouts(fed, xcol, ycol, ax)

        ax.tick_params(axis='x', which='major', labelsize=9)
        ax.tick_params(axis='y', which='major', labelsize=10)
        yformatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        ax.yaxis.set_major_formatter(yformatter)
        ax.set_yticks(np.arange(ymin, ymax, 0.0005))

        # refresh canvas
        self.canvas.draw()
        self.figure.tight_layout(pad=0.5)

    def draw_breakouts(self, df, xcol, ycol, ax):
        """
        """
        df = df.dropna()

        if df.empty:
            return

        start_date = pd.to_datetime(df.BreakoutStartDate.iloc[0])
        end_date = pd.to_datetime(df.BreakoutEndDate.iloc[0])
        start_price = df.BreakoutStartPrice.iloc[0]
        end_price = df.BreakoutEndPrice.iloc[0]
        ax.annotate("Start", xy=(start_date, start_price))
        ax.annotate("End", xy=(end_date, end_price))
        ax.scatter(x=(start_date, end_date), y=(start_price, end_price),
                   s=80, color=[1, 0, 0, 0.75])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ep = EventsPlotter((800, 600))
    ep.show()
    app.exec_()