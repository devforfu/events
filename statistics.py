import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


__all__ = ["Statistics"]


class Statistics:

    def __init__(self, filename, delimiter=',', limit=None):
        self._filename = filename
        df = pd.read_csv(filename, delimiter=delimiter)
        if limit is not None:
            df = df.head(limit)
        df["DateAndTime"] = pd.to_datetime(df["DateAndTime"])
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


def plot_breakouts2(filename):
    """ Helper function for testing purposes.

        Draws found breakouts onto event data plot.
    """
    stat = Statistics(filename)
    df = stat.breakout_calculation(min_change=0.00025, max_pullback=0.0010)

    if not stat.breakouts:
        print("Warning: provided dataset has no breakouts with specified parameters")
        return

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    df = df.dropna()

    for k, g in df.groupby(["DateUTC", "TimeUTC", "Event"]):
        ax = g.plot(x=["DateAndTime"], y=["Ask price"], ax=ax)
        start_date, end_date = g.BreakoutStartDate.iloc[0], g.BreakoutEndDate.iloc[0]
        start_price, end_price = g.BreakoutStartPrice.iloc[0], g.BreakoutEndPrice.iloc[0]
        ax.annotate("Start", xy=(start_date, start_price))
        ax.annotate("End", xy=(end_date, end_price))
        ax.scatter(x=(start_date, end_date), y=(start_price, end_price),
                   s=80, color=[1, 0, 0, 0.75])
        fig = ax.get_figure()
        fig.savefig(" ".join(k).replace(':', '-').replace('/', '') + ".png")
        ax.clear()

    df.to_csv("linked_with_breakouts.csv", index=False)

                
if __name__ == "__main__":
    plot_breakouts2("linked_sample.csv")