""" Gets as input files with events names and ask/bid price changes and links
    them together.
"""
import sys
import argparse
import logging
from operator import methodcaller
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


DATE_AND_TIME_FORMATS = [
    ("%Y.%m.%d %H:%M", "%Y%m%d %H:%M:%S:%f"),
    ("%d.%m.%Y %H:%M", "%Y%m%d %H:%M:%S:%f"),
    ("%Y.%m.%d %H:%M", "%d%m%Y %H:%M:%S:%f"),
    ("%d.%m.%Y %H:%M", "%d%m%Y %H:%M:%S:%f")
]


# TODO: merge utility methods
def convert_date(d, tz):
    """ Utility function to parse date represented in one of predefined
        formats and fix time

        Arguments:
            d (str): date and time as string
    """
    result = None
    for dfmt, _ in DATE_AND_TIME_FORMATS:
        try:
            result = pd.to_datetime(d, format=dfmt)
            result += timedelta(hours=tz, minutes=5)
        except ValueError:
            continue
        break
    if result is None:
        raise ValueError("cannot parse provided date-time string")
    return result


def convert_timestamp(ts):
    """ Utility function to parse time
    """
    result = None
    for _, tfmt in DATE_AND_TIME_FORMATS:
        try:
            result = pd.to_datetime(ts, format=tfmt)
        except ValueError:
            continue
        break
    if result is None:
        raise ValueError("cannot parse provided timestamp string")
    return result


def binary_search(dataframe, attr, cond, low, high):
    while low <= high:
        index = (low + high) // 2
        val = dataframe[attr].iloc[index]
        result = cond(val)
        if result == 0:
            return index
        elif result == -1:
            low = index + 1
        elif result == 1:
            high = index - 1
    # not found
    return -1


class EventsPreprocessor:

    EventsDelimiter = ';'
    DataDelimiter = ','

    def __init__(self, logger=None):
        self.args = self.parse_arguments()
        self.events = self._clean_up(
            pd.read_csv(self.args["events"], delimiter=self.EventsDelimiter))
        self.data = pd.DataFrame()
        self.logger = logger

    @property
    def verbose(self):
        return self.args["verbose"]

    @staticmethod
    def parse_arguments():
        """ Parses passed arguments and returns them as a dictionary
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("-e", "--events", type=str,
                            help="path to events CSV-file")
        parser.add_argument("-d", "--data", type=str,
                            help="path to data CSV-file")
        parser.add_argument("-l", "--limit", nargs='?', type=int, default=None,
                            help="max records to be processed")
        parser.add_argument("-t", "--timezone", nargs='?', type=int, default=5,
                            help="date and time shift")
        parser.add_argument("-o", "--optimized", action='store_true',
                            help="if specified, then data CSV will be processed"
                                 " by small chunks to escape memory issues")
        parser.add_argument("-v", "--verbose", action='store_true')
        return vars(parser.parse_args())

    def log(self, fs, *args, severe=False):
        if not severe and not self.verbose:
            return
        self.logger.debug(fs, *args)

    def run(self, events_chunk=1e2, data_chuck=10e6):
        """ Starts linkage process between events and price data
        """
        if not self.args:
            self.args = self.parse_arguments()

        if self.args["verbose"]:
            if not self.logger:
                raise ValueError("Logger object is not defined")
            sh = logging.StreamHandler(stream=sys.stdout)
            sh.setLevel(logging.DEBUG)
            self.logger.addHandler(sh)

        self.log("[.] Start linkage process...")

        if self.args["optimized"]:
            es, ds = int(events_chunk), int(data_chuck)
            self.process_events_optimised(es, ds)
        else:
            self.process_events()

        self.log("[!] Linkage process finished")

    def process_events(self):
        """ Processes events and data CSV-files.

            All events that have specified Time are linked with appropriate
            data. Each all-day event description is saved into separate
            CSV-file.

            Arguments:
                events (str): path to CSV-file with events
                data (str): path CSV-file with ask/bid data
                limit (int): number of rows to read from file (all if None)
                tz (int): timezone shift (in hours)
        """
        self.data = pd.read_csv(
            self.args["data"], delimiter=self.DataDelimiter, index_col=False)
        df_events, df_data = self.events, self.data

        self.log("[.] Timed and special events separation...")

        indexer = df_events.Time.str.contains("\d\d:\d\d", regex=True, na=False)
        timed_events = df_events[indexer]
        several_days_events = df_events[~indexer]

        self.log("[.] Events and data linking...")

        linked_with_data = self._link_data_and_events(
            timed_events, df_data, self.args["limit"], self.args["timezone"])

        if linked_with_data.empty:
            err = "Error occurred: linked dataframe is empty"
            self.log(err, severe=True)
            raise ValueError(err)

        self.log("[.] Processed dataframes saving...")

        linked_with_data.to_csv("linked.csv", index=False)
        several_days_events.to_csv("special_events.csv", index=False)

    def process_events_optimised(self, events_chuck_size, data_chunk_size):
        """ Enhanced function that used to split big event's prices data file
            into smaller parts to solve memory issues.
        """
        ev = self.events
        tz = self.args["timezone"]
        indexer = ev.Time.str.contains("\d\d:\d\d", regex=True, na=False)
        timed_events, several_days_events = ev[indexer], ev[~indexer]

        self.data = pd.read_csv(self.args["data"],
                                iterator=True, chunksize=data_chunk_size)

        self.log("[.] Events and data linking...")

        start, end = 0, events_chuck_size
        count = 1
        while True:
            events_slice = timed_events.iloc[start:end]

            if events_slice.empty:
                break

            last_date, last_time = events_slice[['Date', 'Time']].iloc[-1]
            upper_bound = convert_date(last_date + " " + last_time, tz)
            relevant_dates = pd.DataFrame()

            for chunk in self.data:
                linked, rest = self._process_chuck(
                    chunk, upper_bound, events_slice, relevant_dates)
                relevant_dates = rest

                if linked is None:
                    continue

                if linked.empty:
                    err = "[!] Warning: linked dataframe is empty"
                    self.log(err, severe=True)
                    continue

                self.log("[.] Events from {} to {} were linked. "
                         "Dataframe size: {}", start + 1, end, )

                filename = 'linked_events_{}_to_{}.csv'.format(start + 1, end)
                linked.to_csv(filename, index=False)

            count += 1
            start = end
            end += events_chuck_size

    def _process_chuck(self, c, upper_bound, events_slice, relevant_dates):
        """ Helper method that links events and data for small piece
            of large dataframe.
        """
        tz = self.args["timezone"]

        self.log("[.] Next data chunk processing...")

        def search_cond(ts):
            """ Used in binary search to split ask/bid prices dataframe
                into appropriately sized chucks with accordance with
                selected events range
            """
            ts = convert_timestamp(ts)
            at = ["year", "month", "day", "hour", "minute", "second"]
            if all(getattr(ts, a) == getattr(upper_bound, a) for a in at):
                return 0
            elif ts < upper_bound:
                return -1
            elif ts > upper_bound:
                return 1

        low, high = 0, len(c.Timestamp) - 1
        idx = binary_search(c, 'Timestamp', search_cond, low, high)
        below, above = c.iloc[:idx], c.iloc[idx:]

        relevant_dates = pd.concat([relevant_dates, below])

        if above.empty:
            return None, relevant_dates

        linked = self._link_data_and_events(
            events_slice, relevant_dates, timezone=tz)

        return linked, above

    @staticmethod
    def _clean_up(ev):
        """ Removes invalid or not interesting rows from dataset
        """
        ev = ev[(ev.Currency == "USD") | (ev.Currency == "EUR")]
        ev = ev[ev.Importance == "H"]
        for text_column in ['Currency', 'Importance', 'Event']:
            ev[text_column] = ev[text_column].str.strip()
        return ev

    @staticmethod
    def _link_data_and_events(ev, data, limit=None, timezone=None):
        """ Creates data frame with events and data linked.

            Arguments:
                ev (DateFrame): path CSV-file with ask/bid data
                data (DataFrame): path to CSV-file with events
                limit (int): quantity of data rows to be processed
                timezone (int): amount of hours to shift date columns

            Returns:
                DataFrame: pandas dataframe with events and data linked
        """
        if limit is not None:
            data = data.head(limit)

        date, time = ev["Date"], ev["Time"]

        for dfmt, tfmt in DATE_AND_TIME_FORMATS:
            try:
                ev["DateAndTime"] = [pd.to_datetime(d + " " + t, format=dfmt)
                                     for d, t in zip(date, time)]
                data["DateAndTime"] = [pd.to_datetime(ts, format=tfmt)
                                       for ts in data["Timestamp"]]
            except ValueError:
                continue
            date_time_format = dfmt
            break

        if timezone is not None:
            ev.DateAndTime += timedelta(hours=timezone)

        output = list()
        # group events with proper data
        for event in ev.values:
            ts = event[-1]
            one_min_before = ts - timedelta(minutes=1)
            five_min_ahead = ts + timedelta(minutes=5)
            data_in_range = data[(data.DateAndTime >= one_min_before) &
                                 (data.DateAndTime <= five_min_ahead)]

            for d in data_in_range.values:
                output.append(np.hstack((event[:-1], d)))

        if not output:
            return pd.DataFrame()

        cols = list(ev.columns[:-1]) + list(data.columns)
        dataframe = pd.DataFrame(output, columns=cols)

        # synchronize original Date and Time columns with DateAndTime
        if timezone is not None:
            dates = pd.Series(datetime.strptime(d + " " + t, date_time_format)
                              + timedelta(hours=timezone)
                              for d, t in zip(dataframe.Date, dataframe.Time))
            dataframe["DateUTC"] = dates.map(methodcaller('date'))
            dataframe["TimeUTC"] = dates.map(methodcaller('time'))

        return dataframe


def init_logger():
    logger = logging.getLogger("events_log")
    fh = logging.FileHandler("events.log", mode='a')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    return logger


if __name__ == '__main__':
    ep = EventsPreprocessor(logger=init_logger())
    ep.run()