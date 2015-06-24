import os
import math
import shutil
import logging
import argparse
from operator import itemgetter, ge, le
from datetime import datetime, timedelta
from multiprocessing import Queue, Process

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


__all__ = ["Statistics", "calculate_statistics_threaded"]

logging.basicConfig(filename="statistics.log", level=logging.DEBUG)


def take(seq, n, default=None):
    if len(seq) >= n:
        return seq[:n]
    ls = seq
    diff = n - len(seq)
    while diff:
        ls.append(default)
        diff -= 1
    return ls


class Statistics:

    def __init__(self, filename=None, dataframe=None, delimiter=',',
                 limit=None, verbose=True):
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
        self._verbose = verbose
        self._breakouts = dict()

    @property
    def breakouts(self):
        return self._breakouts

    def breakout_calculation(self, attr='Ask price', min_change=0.0020,
                             max_duration=10, max_pullback=0.0010):
        """ Calculates brakeouts for linked events.

            Breakout is defined by two points (start_date, start_price)
            and (end_date, end_price), price delta and distance from
            event. Found breakouts are stored into file with linked events.
        """
        df = self._df
        verbose = self._verbose
        index = dkey, pkey = ["DateAndTime", attr]
        dt = timedelta(seconds=max_duration)

        for key, ev in df.groupby(["DateUTC", "TimeUTC", "Event"]):
            if verbose:
                d, t, name = key
                logging.info("[*] Process event '%s' ...", name)

            dates = ev[dkey]
            breakout_start_point, breakout_end_point = None, None
            min_change_point = None
            date_to_continue = None

            for date in dates:                                
                sf = ev[(ev[dkey] >= date) & (ev[dkey] <= date + dt)]  # i.e. sub-frame
                imin, imax = sf[pkey].idxmin(), sf[pkey].idxmax()
                values = sf[index].ix[[imin, imax]].values
                (dmin, pmin), (dmax, pmax) = take(values, 2)
                
                price_diff = abs(pmax - pmin)
                if price_diff < min_change:
                    continue
                    
                # ask price at the end of inspected 10 seconds interval
                last = sf.iloc[-1][pkey]

                # at this point we have identified breakout
                breakout_start_point = (dmin, pmin) if dmin <= dmax else (dmax, pmax)
                min_change_point = (dmax, pmax) if dmin < dmax else (dmin, pmin)
                date_to_continue = date + dt

                break  # only the first breakout is needed

            if breakout_start_point is None:
                # breakout was not identified - skip to next event
                continue

            date, price = min_change_point

            for _, d, p in ev[ev[dkey] >= date][[dkey, pkey]].itertuples():
                # breakout from higher to lower value
                if dmax < dmin:
                    if p < price:
                        date, price = d, p
                    elif abs(p - price) >= max_pullback:
                        breakout_end_point = d, p
                        break

                # breakout from lower to higher value
                else:
                    if p > price:
                        date, price = d, p
                    elif abs(p - price) >= max_pullback:
                        breakout_end_point = d, p
                        break

            if breakout_end_point is None:
                breakout_end_point = d, t = ev[[dkey, pkey]].iloc[-1].values

            if verbose:
                logging.info("[+] Breakout was found")

            self._breakouts[key] = breakout_start_point, breakout_end_point

        for key, breakout in self._breakouts.items():
            d, t, name = key
            try:
                condition = (df.DateUTC == d) & (df.TimeUTC == t) & (df.Event == name)
                (start_date, start_price), (end_date, end_price) = breakout
            except (TypeError, ValueError):
                logging.warning("[-] Breakout processing issue for %s event", name)
                continue
            diff = abs(end_price - start_price)
            event_date = pd.to_datetime(d + " " + t)
            if event_date > start_date:
                start_diff = event_date - start_date
            else:
                start_diff = start_date - event_date
            if event_date > end_date:
                end_diff = event_date - end_date
            else:
                end_diff = end_date - event_date

            start_diff = start_diff.to_pytimedelta().seconds
            end_diff = end_diff.to_pytimedelta().seconds
            df.ix[condition, "BreakoutStartDate"] = start_date
            df.ix[condition, "BreakoutEndDate"] = end_date
            df.ix[condition, "BreakoutStartPrice"] = start_price
            df.ix[condition, "BreakoutEndPrice"] = end_price
            df.ix[condition, "BreakoutPriceDelta"] = round(diff, 5)
            df.ix[condition, "DistanceFromStartToEvent"] = start_diff
            df.ix[condition, "DistanceFromEndToEvent"] = end_diff
        return df


def collect_breakouts(df):
    """ Extract all breakouts info from data frame """
    index = ["Event", "DateUTC", "TimeUTC", "BreakoutStartDate",
             "BreakoutStartPrice", "BreakoutEndDate", "BreakoutEndPrice",
             "BreakoutPriceDelta", "DistanceFromStartToEvent",
             "DistanceFromEndToEvent"]
    df = df.dropna()
    breakouts, filtered = df[index], list()
    for key, group in breakouts.groupby(["Event", "DateUTC", "TimeUTC"]):
        filtered.append(group.iloc[0].values)
    filtered_dataframe = pd.DataFrame(filtered, columns=index)
    return filtered_dataframe


def calc_thread(queue, order, df, stat_params):
    """ Daemon thread for linked events statistics calculation """
    stat = Statistics(dataframe=df)
    try:
        processed = stat.breakout_calculation(**stat_params)
    except Exception as e:
        err = "[-] Thread #%d: unexcpected error occurred - %s"
        logging.error(err, order, str(e))
    else:
        queue.put((order, processed))


def calculate_statistics_threaded(input_file, output_file=None,
                                  min_change=0.0025, max_pullback=0.0010):
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
    params = dict(min_change=min_change, max_pullback=max_pullback)

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
    return merged_frame


def plot_breakouts(filename, args):
    """ Helper function for testing purposes.

        Draws found breakouts onto event data plot.
    """
    df = pd.read_csv(filename)
    df["DateAndTime"] = pd.to_datetime(df.DateAndTime)
    df = df.dropna()

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    xcol, ycol = "DateAndTime", "Ask price"

    subfolder = "min_change_%.6f_max_pullback_%.6f" % \
              (args["min_change"], args["max_pullback"])
    image_folder = os.path.join("img", subfolder)
    os.makedirs(image_folder)

    for k, g in df.groupby(["DateUTC", "TimeUTC", "Event"]):
        ax = g.plot(x=xcol, y=[ycol], ax=ax)
        ymin = math.ceil(g[ycol].min().min() * 1000) / 1000.0 - 0.0005
        ymax = math.ceil(g[ycol].max().max() * 1000) / 1000.0 + 0.0005
        start_date, end_date = g.BreakoutStartDate.iloc[0], g.BreakoutEndDate.iloc[0]
        start_price, end_price = g.BreakoutStartPrice.iloc[0], g.BreakoutEndPrice.iloc[0]

        ax.annotate("Start", xy=(start_date, start_price))
        ax.annotate("End", xy=(end_date, end_price))
        ax.scatter(x=(start_date, end_date), y=(start_price, end_price),
                   s=80, color=[1, 0, 0, 0.75])

        yformatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        ax.yaxis.set_major_formatter(yformatter)
        ax.set_yticks(np.arange(ymin, ymax, 0.0005))

        fig = ax.get_figure()
        image_name = " ".join(k).replace(':', '-').replace('/', '') + ".png"
        fig.savefig(os.path.join(image_folder, image_name))
        logging.debug("[+] Image %s saved into '%s' folder" %
                      (image_name, image_folder))
        ax.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=str,
                        help="file with linked events names")
    parser.add_argument("-o", "--output", type=str,
                        help="file with breakouts name")
    parser.add_argument("-t", "--threaded", action='store_true',
                        help="if true then statistics will be calculated with multiprocessing")
    parser.add_argument("--min-change", type=float, action='store', default=0.0025,
                        help="minimum change to identify breakout start")
    parser.add_argument("--max-pullback", type=float, action='store', default=0.0010,
                        help="maximum pullback to identify breakout end")
    parser.add_argument("--breakouts-only", action='store_true',
                        help="store breakouts only (without price values data)")
    args = vars(parser.parse_args())

    input_file, output_file = args["input"], args["output"]
    min_change, max_pullback = args["min_change"], args["max_pullback"]
    breakouts_only = args["breakouts_only"]

    if args["threaded"]:
        df = calculate_statistics_threaded(
            input_file, output_file, min_change, max_pullback)
    else:
        stat = Statistics(filename=input_file)
        df = stat.breakout_calculation(
            min_change=min_change, max_pullback=max_pullback)

    folder = os.path.dirname(output_file)
    if not os.path.exists(folder):
        os.makedirs(folder)
    df.to_csv(output_file, index=False)

    # for debug purposes only
    plot_breakouts(output_file, args)

    if breakouts_only:
        breakouts = collect_breakouts(df)
        # rewrite previous file
        breakouts.to_csv(output_file)