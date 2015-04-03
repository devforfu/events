import os
from operator import itemgetter
from datetime import datetime, timedelta
from multiprocessing import Queue, Process

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


__all__ = ["Statistics", "calculate_statistics_threaded"]


class Statistics:

    def __init__(self, filename=None, dataframe=None, delimiter=',', limit=None):
        if not filename and not isinstance(dataframe, pd.DataFrame):
            raise ValueError("filename or dataframe should be defined")

        if filename and isinstance(dataframe, pd.DataFrame):
            raise ValueError("filename and dataframe cannot be defined both")

        if dataframe is not None:
            df = dataframe
        else:
            df = pd.read_csv(filename, delimiter=delimiter)

        df["DateAndTime"] = pd.to_datetime(df.DateAndTime)

        if limit is not None:
            df = df.head(limit)
        self._df = df
        self._breakouts = dict()

    @property
    def breakouts(self):
        return self._breakouts

    def breakout_calculation(self, attr='Ask price', min_change=0.0020,
                             max_duration=10, max_pullback=0.0010):
        """ Calculates brakeouts for linked events.
        """
        df = self._df
        index = dkey, pkey = ["DateAndTime", attr]
        dt = timedelta(seconds=max_duration)

        for key, ev in df.groupby(["DateUTC", "TimeUTC", "Event"]):
            dates = ev[dkey]

            for date in dates:                                
                sf = ev[(ev[dkey] >= date) & (ev[dkey] <= date + dt)]  # i.e. sub-frame
                imin, imax = sf[pkey].idxmin(), sf[pkey].idxmax()                
                (dmin, pmin), (dmax, pmax) = sf[index].ix[[imin, imax]].values
                
                price_diff = abs(pmax - pmin)
                if price_diff < min_change:
                    continue
                    
                # ask price at the end of inspected 10 seconds interval
                last = sf.iloc[-1][pkey]
                
                # difference between breakout ending (min or max) value 
                # and last value from considered interval
                pullback = abs(last - (pmax if dmin < dmax else pmin))
                if pullback > max_pullback:
                    continue

                self._breakouts[key] = dict(dmin=dmin, pmin=pmin, dmax=dmax, pmax=pmax)

                break  # only first breakout is needed

        for key, breakout in self._breakouts.items():
            print(key)
            d, t, name = key
            condition = (df.DateUTC == d) & (df.TimeUTC == t) & (df.Event == name)
            dmin, dmax = breakout["dmin"], breakout["dmax"]
            pmin, pmax = breakout["pmin"], breakout["pmax"]

            df.ix[condition, "BreakoutStartDate"] = dmin if dmin < dmax else dmax
            df.ix[condition, "BreakoutEndDate"] = dmin if dmin > dmax else dmax
            df.ix[condition, "BreakoutStartPrice"] = pmin if dmin < dmax else pmax
            df.ix[condition, "BreakoutEndPrice"] = pmin if dmin > dmax else pmax
            df.ix[condition, "BreakoutPriceDelta"] = round(abs(pmin - pmax), 5)

        return df


def calc_thread(queue, order, df, stat_params):
    """ Daemon thread for linked events statistics calculation """
    stat = Statistics(dataframe=df)
    processed = stat.breakout_calculation(**stat_params)
    queue.put((order, processed))


def calculate_statistics_threaded(input_file, output_file=None):
    """ Multiprocessing events statistic calculation.

        Splits initial dataframe into several almost equal subframes
        and calculates statistics for each of them.
    """
    if output_file is None:
        output_file = input_file

    df = pd.read_csv(input_file)
    df["DateAndTime"] = pd.to_datetime(df["DateAndTime"])

    n = os.cpu_count()
    queue = Queue()
    procs = list()
    params = dict(min_change=0.0020, max_pullback=0.0010)

    event_names = [name for (_, _, name), g in df.groupby(["DateUTC", "TimeUTC", "Event"])]

    for i, names_range in enumerate(np.array_split(event_names, n, axis=0)):
        sf = pd.DataFrame()
        for name in names_range:
            sf = pd.concat([sf, df[df.Event == name]])
        p = Process(target=calc_thread, args=(queue, i, sf, params))
        p.daemon = True
        procs.append(p)

    for p in procs:
        p.start()

    processed = list()
    while True:
        try:
            order, subframe = queue.get(timeout=3600)
        except Queue.Empty:
            break
        processed.append((order, subframe))
        if len(processed) == n:
            break

    sorted(processed, key=itemgetter(0))
    merged_frame = pd.concat([frame for _, frame in processed])
    merged_frame.to_csv(output_file, index=False)


def plot_breakouts(filename):
    """ Helper function for testing purposes.

        Draws found breakouts onto event data plot.
    """
    df = pd.read_csv(filename)
    df["DateAndTime"] = pd.to_datetime(df.DateAndTime)
    df = df.dropna()

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)

    for k, g in df.groupby(["DateUTC", "TimeUTC", "Event"]):
        ax = g.plot(x="DateAndTime", y=["Ask price"], ax=ax)
        start_date, end_date = g.BreakoutStartDate.iloc[0], g.BreakoutEndDate.iloc[0]
        start_price, end_price = g.BreakoutStartPrice.iloc[0], g.BreakoutEndPrice.iloc[0]
        ax.annotate("Start", xy=(start_date, start_price))
        ax.annotate("End", xy=(end_date, end_price))
        ax.scatter(x=(start_date, end_date), y=(start_price, end_price),
                   s=80, color=[1, 0, 0, 0.75])
        fig = ax.get_figure()
        fig.savefig(" ".join(k).replace(':', '-').replace('/', '') + ".png")
        ax.clear()


if __name__ == "__main__":
    input_file, output_file = "linked.csv", "linked_with_breakout.csv"

    calculate_statistics_threaded(input_file, output_file)

    # stat = Statistics(filename=input_file)
    # df = stat.breakout_calculation(min_change=0.00025, max_pullback=0.0010)
    # df.to_csv(output_file, index=False)

    plot_breakouts(output_file)
