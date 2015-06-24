""" Gets as input files with events names and ask/bid price changes and links
    them together.
"""
import os
import sys
import argparse
import logging
import pathlib
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
MEMORY_LIMIT = 2.0

# get rid off false pandas warning
pd.options.mode.chained_assignment = None 


# TODO: merge utility methods
def convert_date(d, tz):
    # result = None
    # for dfmt, _ in DATE_AND_TIME_FORMATS:
    #     try:
    #         result = pd.to_datetime(d, format=dfmt)
    #         result += timedelta(hours=tz, minutes=5)
    #     except ValueError:
    #         continue
    #     break
    # if result is None:
    #     raise ValueError("cannot parse provided date-time string")
    # return result
    return convert(d, mode='date')


def convert_timestamp(ts):
    # result = None
    # for _, tfmt in DATE_AND_TIME_FORMATS:
    #     try:
    #         result = pd.to_datetime(ts, format=tfmt)
    #     except ValueError:
    #         continue
    #     break
    # if result is None:
    #     raise ValueError("cannot parse provided timestamp string")
    # return result
    return convert(ts, mode='data')


def convert(value, mode='date'):
    """ Utility function to parse date represented in one of predefined
        formats and fix time

        Arguments:
            value (str): datetime or timestamp as a string

        Returns:
            (datetime): parsed data
    """
    import operator
    result = None
    if mode == 'date':
        get_item = operator.itemgetter(0)
    elif mode == 'timestamp':
        get_item = operator.itemgetter(1)
    else:
        raise ValueError('unexpected mode')
    for template in DATE_AND_TIME_FORMATS:
        try:
            result = pd.to_datetime(value, format=get_item(template))
        except ValueError:
            continue
        break
    if result is None:
        err = "cannot parse provided "
        raise ValueError(err + ('date' if mode == 'date' else 'timestamp'))
    return result


def binary_search(dataframe, attr, cond, low, high):
    """ Generic binary search implementation.

        Proceeds with search among specified dataframe attribute values
        until condition function does not return zero value or low index
        become greater then high.
    """
    val = None
    while low <= high:
        index = (low + high) // 2
        val = dataframe[attr].iloc[index]
        result = cond(val)
        if result == 0:
            return True, index
        elif result == -1:
            low = index + 1
        elif result == 1:
            high = index - 1
    # not found
    return False, val


class EventsPreprocessor:

    EventsDelimiter = ';'
    DataDelimiter = ','

    def __init__(self, logger=None):
        self.args = self.parse_arguments()
        if self.args["events"] is None or self.args["data"] is None:
            raise ValueError("events or data file is not specified")
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
        parser.add_argument("--output-folder", nargs='?', type=str,
                            default="linked")
        return vars(parser.parse_args())

    def log(self, fs, *args, severe=False):
        if not severe and not self.verbose:
            return
        self.logger.debug(fs, *args)

    def run(self, events_chunk=1e2, data_chuck=10e6):
        """ Starts linkage process between events and price data.
        
            Arguments:
                events_chunk (int): quantity of events dataframe rows
                    processed per iteration (optimized mode only)
                data_chunk (int): quantity of events data dataframe rows  
                    processed per iteration (optimized mode only)
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

        output_folder = self.args["output_folder"]
        if os.path.exists(output_folder):
            self.log("[!] Warning: output folder already exists. ", severe=True)
            os.rmdir(output_folder)

        os.mkdir(output_folder)

        if self.args["optimized"]:
            es, ds = int(events_chunk), int(data_chuck)
            self.process_events_optimised(es, ds)
        else:
            file_size = pathlib.Path(self.args["data"]).stat().st_size
            gigabyte = 1024 ** 3
            if file_size / gigabyte > MEMORY_LIMIT:
                self.log("[!] Warning: specified data CSV file size "
                         "is greater then 2 GB. Try to use -o flag "
                         "if memory issues occur", severe=True)
            self.process_events()

        self.log("[!] Linkage process finished")
        self.log("[.] Output folder: %s" % output_folder)

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

        # linked_with_data.to_csv("linked.csv", index=False)
        # several_days_events.to_csv("special_events.csv", index=False)
        folder = self.args["output_folder"]
        linked_with_data.to_csv(os.path.join(folder, "linked.csv"), index=False)
        several_days_events.to_csv(
            os.path.join(folder, "special_events.csv"), index=False)

    def process_events_optimised(self, events_chuck_size, data_chunk_size):
        """ Enhanced function that used to split big event's prices data file
            into smaller parts to solve memory issues.
        """
        ev = self.events
        tz = self.args["timezone"]
        indexer = ev.Time.str.contains("\d\d:\d\d", regex=True, na=False)
        timed_events, several_days_events = ev[indexer], ev[~indexer]
                
        if not several_days_events.empty:
            several_days_events.to_csv("special_events.csv", index=False)
            self.log("[+] Special events were saved into standalone CSV-file")
        else:
            self.log("[!] Special events not found")

        self.data = pd.read_csv(self.args["data"],
                                iterator=True, chunksize=data_chunk_size)

        self.log("[.] Events and data linking...")

        start, end = 0, events_chuck_size
        relevant_dates = pd.DataFrame()
        count = 1
        while True:
            events_slice = timed_events.iloc[start:end]
            # TODO: remove in release version
            # events_slice.to_csv('slice_{}_{}.csv'.format(start, end),
            #                     index=False)

            if events_slice.empty:
                break

            first_date, first_time = events_slice[['Date', 'Time']].iloc[0]
            lower_bound = convert(first_date + " " + first_time, mode='date')
            lower_bound += timedelta(hours=tz, minutes=-1)

            last_date, last_time = events_slice[['Date', 'Time']].iloc[-1]
            upper_bound = convert(last_date + " " + last_time, mode='date')
            upper_bound += timedelta(hours=tz, minutes=5)
            
            self.log("[.] Events slice bounded by [%s; %s] is in processing...",
                     lower_bound, upper_bound)
            
            for chunk in self.data:
                bounds = (lower_bound, upper_bound)
                linked, rest = self._process_chuck(
                    chunk, bounds, events_slice, relevant_dates)

                relevant_dates = rest

                if linked is None:
                    if relevant_dates.empty:
                        err = "[!] Warning: events from %d to %d have no data"
                        self.log(err, start + 1, end)
                        break
                    else:
                        continue

                if linked.empty:
                    err = "[!] Warning: linked dataframe is empty"
                    self.log(err, severe=True)
                    continue

                self.log("[+] Events from %d to %d were linked. "
                         "Dataframe size: %d", start + 1, end, linked.shape[0])

                filename = 'linked_events_{}_to_{}.csv'.format(start + 1, end)
                filename = os.path.join(self.args["output_folder"], filename)
                linked.to_csv(filename, index=False)
                linked = pd.DataFrame()
                break

            count += 1
            start = end
            end += events_chuck_size

    def _process_chuck(self, c, bounds, events_slice, relevant_dates):
        """ Helper method that links events and data for small piece
            of large dataframe.
        """
        lower_bound, upper_bound = bounds
        tz = self.args["timezone"]

        # current data chunk has no data for selected events slice
        last_timestamp = convert(c.Timestamp.iloc[-1], mode='timestamp')
        too_early_data = lower_bound > last_timestamp
        first_timestamp = convert(c.Timestamp.iloc[0], mode='timestamp')
        too_late_data = upper_bound < first_timestamp

        if too_early_data or too_late_data:
            return None, pd.DataFrame()

        def search_cond(ts):
            """ Used in binary search to split ask/bid prices dataframe
                into appropriately sized chucks with accordance with
                selected events range
            """
            ts = convert(ts, mode='timestamp')
            at = ["year", "month", "day", "hour", "minute"]
            if all(getattr(ts, a) == getattr(upper_bound, a) for a in at):
                return 0
            elif ts < upper_bound:
                return -1
            elif ts > upper_bound:
                return 1

        low, high = 0, len(c.Timestamp) - 1
        ok, idx = binary_search(c, 'Timestamp', search_cond, low, high)

        if ok:
            below, above = c.iloc[:idx], c.iloc[idx:]
        else:
            below, above = c, pd.DataFrame()
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

    def _link_data_and_events(self, ev, data, limit=None, timezone=None):
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

        self.log("[.] Data filtering...")

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

        self.log("[.] Timezones synchronization...")

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
    ep.run(events_chunk=50, data_chuck=1e6)
