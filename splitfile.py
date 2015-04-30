import argparse
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

    def __init__(self):
        self.args = self.parse_arguments()
        self.events = self._clean_up(
            pd.read_csv(self.args["events"], delimiter=self.EventsDelimiter))
        self.data = pd.DataFrame()

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument("-e", "--events", type=str,
                            help="path to events CSV-file")
        parser.add_argument("-d", "--data", type=str,
                            help="path to data CSV-file")
        parser.add_argument("-l", "--limit", nargs='?', type=int, default=None,
                            help="max records to be processed")
        parser.add_argument("-t", "--timezone", nargs='?', type=int, default=5,
                            help="date and time shift")
        return vars(parser.parse_args())

    def run(self, opt=False):
        """ Starts linkage process between events and price data
        """
        if not self.args:
            self.parse_arguments()

        if opt:
            es, ds = int(1e2), int(10e6)
            process = lambda: self.process_events_optimised(es, ds)
        else:
            process = self.process_events

        linked, special = process()
        if linked.empty:
            raise ValueError("Error occurred: dataframe is empty")
        linked.to_csv("linked.csv", index=False)
        special.to_csv("special_events.csv", index=False)

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

            Returns:
                linked_with_data (DataFrame):
                several_days_events (DataFrame):
        """
        self.data = pd.read_csv(
            self.args["data"], delimiter=self.DataDelimiter, index_col=False)
        df_events, df_data = self.events, self.data
        indexer = df_events.Time.str.contains("\d\d:\d\d", regex=True, na=False)
        timed_events = df_events[indexer]
        several_days_events = df_events[~indexer]

        linked_with_data = self._link_data_and_events(
            timed_events, df_data, self.args["limit"], self.args["timezone"])

        return linked_with_data, several_days_events

    def process_events_optimised(self, events_chuck_size, data_chunk_size):
        """ Enhanced function that used to split big event's prices data file
            into smaller parts to solve memory issues.
        """
        ev = self.events
        indexer = ev.Time.str.contains("\d\d:\d\d", regex=True, na=False)
        timed_events, several_days_events = ev[indexer], ev[~indexer]

        self.data = pd.read_csv(self.args["data"],
                                iterator=True, chunksize=data_chunk_size)
        tz = self.args["timezone"]

        start, end = 0, events_chuck_size
        count = 1
        while True:
            events_slice = timed_events.iloc[start:end]
            last_date, last_time = events_slice[['Date', 'Time']].iloc[-1]
            upper_bound = convert_date(last_date + " " + last_time, tz)
            relevant_dates = pd.DataFrame()

            def search_cond(ts):
                ts = convert_timestamp(ts)
                at = ["year", "month", "day", "hour", "minute"]
                if all(getattr(ts, a) == getattr(upper_bound, a) for a in at):
                    return 0
                elif ts < upper_bound:
                    return -1
                elif ts > upper_bound:
                    return 1

            for chunk in self.data:
                low, high = 0, len(chunk.Timestamp) - 1
                idx = binary_search(chunk, 'Timestamp', search_cond, low, high)
                below, above = chunk.iloc[0:idx], chunk.iloc[idx:]

                relevant_dates = pd.concat([relevant_dates, below])

                if above.empty:
                    continue

                linked = self._link_data_and_events(
                    events_slice, relevant_dates, timezone=tz)
                relevant_dates = above
                linked.save('linked_{}.csv'.format(count))

            count += 1
            start = end
            end += events_chuck_size

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


if __name__ == '__main__':
    ep = EventsPreprocessor()
    ep.run(opt=True)