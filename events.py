import re
import argparse
from datetime import datetime, timedelta
from operator import methodcaller

import numpy as np
import pandas as pd


def process_events(events, data, limit=None, timezone=None):
    """ Processes events and data CSV-files.

        All events that have specified Time are linked with appropriate data.
        Each all-day event description is saved into separate CSV-file.

        Arguments:
            events (str): path CSV-file with ask/bid data
            data (str): path to CSV-file with events

        Returns:
            linked_with_data (DataFrame):
            several_days_events (DataFrame):
    """
    df_events = pd.read_csv(events, delimiter=';', index_col=False)
    df_events = df_events[(df_events.Currency == "USD") | (df_events.Currency == "EUR")]
    df_events = df_events[df_events.Importance == "H"]

    df_data = pd.read_csv(data, index_col=False)

    for text_column in ['Currency', 'Importance', 'Event']:
        df_events[text_column] = df_events[text_column].str.strip()

    indexer = df_events.Time.str.contains("\d\d:\d\d", regex=True, na=False)
    timed_events, several_days_events = df_events[indexer], df_events[~indexer]

    linked_with_data = link_data_and_events(timed_events, df_data, limit, timezone)

    return linked_with_data, several_days_events


def link_data_and_events(ev, data, limit=None, timezone=None):
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
    date_time_format = "%Y.%m.%d %H:%M"
    timestamp_format = "%Y%m%d %H:%M:%S:%f"

    ev["DateAndTime"] = [pd.to_datetime(d + " " + t, format=date_time_format)
                         for d, t in zip(date, time)]

    data["DateAndTime"] = [pd.to_datetime(ts, format=timestamp_format)
                           for ts in data["Timestamp"]]

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
    dates = pd.Series(datetime.strptime(d + " " + t, date_time_format) + timedelta(hours=timezone)
                      for d, t in zip(dataframe.Date, dataframe.Time))

    dataframe["DateUTC"] = dates.map(methodcaller('date'))
    dataframe["TimeUTC"] = dates.map(methodcaller('time'))

    return dataframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--events", type=str, help="path to events CSV-file")
    parser.add_argument("-d", "--data", type=str, help="path to data CSV-file")
    parser.add_argument("-l", "--limit", nargs='?', type=int, default=None,
                        help="max records to be processed")
    parser.add_argument("-t", "--timezone", nargs='?', type=int, default=7,
                        help="date and time shift")
    args = vars(parser.parse_args())

    linked, special = process_events(**args)

    if linked.empty:
        raise ValueError("Error occurred: dataframe is empty")

    linked.to_csv("linked.csv", index=False)
    special.to_csv("special_events.csv", index=False)