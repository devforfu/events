""" Auxiliary module that used to split big event's prices data file into
    smaller parts to solve memory issues.
"""

from datetime import timedelta
import pandas as pd
from events import DATE_AND_TIME_FORMATS, link_data_and_events


def convert_date(d, tz):
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


def split_data_file(events, data, events_chuck_size, data_chunk_size, tz):
    """
    """
    events = pd.read_csv(events, delimiter=';', index_col=False)
    events = events[(events.Currency == "USD") | (events.Currency == "EUR")]
    events = events[events.Importance == "H"]
    for text_column in ['Currency', 'Importance', 'Event']:
        events[text_column] = events[text_column].str.strip()

    indexer = events.Time.str.contains("\d\d:\d\d", regex=True, na=False)
    timed_events, several_days_events = events[indexer], events[~indexer]

    start, end = 0, events_chuck_size

    prices = pd.read_csv(data, iterator=True, chunksize=data_chunk_size)

    count = 1
    while True:
        events_slice = timed_events.iloc[start:end]
        last_date, last_time = events_slice[['Date', 'Time']].iloc[-1]
        upper_bound = convert_date(last_date + " " + last_time, tz)
        relevant_dates = pd.DataFrame()

        for chunk in prices:
            chunk['Timestamp'] = chunk.Timestamp.map(convert_timestamp)
            cond = chunk.Timestamp <= upper_bound
            below, above = chunk[cond], chunk[~cond]
            relevant_dates = pd.concat([relevant_dates, below])

            if above.empty:
                continue

            linked = link_data_and_events(
                events_slice, relevant_dates, timezone=tz)
            relevant_dates = above
            linked.save('linked_{}.csv'.format(count))

        count += 1
        start = end
        end += events_chuck_size


if __name__ == '__main__':
    events, data = 'csv\\HistoryEventsAll_Jan.csv', 'csv\\EURUSD_Jan.csv'
    split_data_file(events, data, 100, 100000, 5)