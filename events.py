import datetime
import numpy as np
import pandas as pd


def link_data_and_events(events_file, data_file, limit=None, add_hours=None):
    """ Creates data frame with events and linked.

        Arguments:
            data_file (str): path CSV-file with ask/bid data
            events_file (str): path to CSV-file with events
            limit (int): quantity of data rows to be processed

        Returns:
            DataFrame: pandas dataframe with events and data linked
    """
    ev = pd.read_csv(events_file, delimiter=";")
    data = pd.read_csv(data_file)
    if limit is not None:
        data = data.head(limit)

    ev = ev[(ev.Currency == "USD") | (ev.Currency == "EUR")]
    ev = ev[ev.Importance == "H"]
    # ev = ev.drop(["Unnamed: " + str(i) for i in range(5, 10)], axis=1)
    date, time = ev["Date"], ev["Time"]

    ev["DateAndTime"] = [pd.to_datetime(d + " " + t, format="%d.%m.%Y %H:%M")
                         for d, t in zip(date, time)]

    data["DateAndTime"] = [pd.to_datetime(ts, format="%Y%m%d %H:%M:%S:%f")
                           for ts in data["Timestamp"]]

    if add_hours is not None:
        ev.DateAndTime += datetime.timedelta(hours=add_hours)
        data.DateAndTime += datetime.timedelta(hours=add_hours)

    output = list()

    # group events with proper data
    for event in ev.values:
        ts = event[-1]
        one_min_before = ts - datetime.timedelta(minutes=1)
        five_min_ahead = ts + datetime.timedelta(minutes=5)
        data_in_range = data[(data.DateAndTime >= one_min_before) &
                             (data.DateAndTime <= five_min_ahead)]

        for d in data_in_range.values:
            output.append(np.hstack((event[:-1], d)))

    cols = list(ev.columns[:-1]) + list(data.columns)
    return pd.DataFrame(output, columns=cols)


if __name__ == "__main__":
    df = link_data_and_events("HistoryEventsAll_Jan.csv", "EURUSD_Jan.csv",
                              limit=500000, add_hours=7)
    df.to_csv("linked.csv", index=False)