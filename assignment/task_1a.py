from datetime import timedelta

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.axis import Axis

from assignment.data_loader import DataLoader
import matplotlib.pyplot as plt

class Task1A:
    """
    Notice all sorts of properties of the dataset: how many records are there, how many
    attributes, what kinds of attributes are there, ranges of values, distribution of values,
    relationships between attributes, missing values, and so on. A table is often a suitable
    way of showing such properties of a dataset. Notice if something is interesting (to you,
    or in general), make sure you write it down if you find something worth mentioning.


    Make various plots of the data. Is there something interesting worth reporting?
    Report the figures, discuss what is in them. What meaning do those bars, lines, dots, etc.
    convey? Please select essential and interesting plots for discussion, as you have limited
    space for reporting your findings.

    Deliverables:
        -> n records
        -> n attributes
        -> kinds of attributes
        -> ranges of values
        -> distribution of values
        -> relationships between attributes
        -> missing values
        -> make various plots of the data

    """

    df = DataLoader.load_to_df()

    @classmethod
    def text(cls):
        """
        -> n records
        -> n attributes
        -> kinds of attributes
        -> ranges of values
        """
        df = cls.df.copy()

        print(f"n records: {len(df)}")
        print("attributes: [id (string), time (datetime), variable (string), value (float)]")
        n_users = len(df.id.unique())
        print(f"we have {n_users} user IDs")
        dates = sorted(df.date.unique())
        min_date, max_date = dates[0], dates[-1]
        print(f"the data spans over {len(dates)} days, from {min_date} to {max_date}")
        print("variables:")
        print(df.groupby("variable").value.agg(["count", "min", "max", "mean", "median"]).round(3).to_string())

    @classmethod
    def variables_distribution_of_values(cls):
        fig, axes = plt.subplots(len(cls.df.date.unique()) // 4, 4, figsize=(12, 10))
        axfl = axes.flatten()

        for (group, sdf), ax in zip(cls.df.groupby("variable"), axfl):
            # ax.hist(sdf.value, bins=30, alpha=0.7, color='b')  # Histogram (normalized)
            ax.plot(sdf.value)
            ax.set_title(group)
            ax.grid(True)
        plt.tight_layout()
        plt.show()


    @classmethod
    def plots(cls):
        """
        plot 1:
            - n records per person per day and total N records bar:
                shows that there was leading and trailing dates where very little was recorded, shows that there
                was a lot of noise per user on recording levels, and that no user

        """

        df = cls.df.groupby(["id", "date"]).apply(len).unstack(0).fillna(0)
        plt.plot(df.rolling(5).mean(), label=df.columns)
        plt.legend()
        plt.title("n_records per person per day")
        plt.show()
        plt.bar([c.removeprefix("AS14.") for c in df.columns], df.sum())
        plt.xticks(rotation=45)
        plt.show()

        # fig, axes = plt.subplots(5, 4, figsize=(12, 10))
        # for (group, sdf), ax in zip(cls.df.groupby("variable"), axes.flatten()):
        #     df = sdf.groupby(["id", "date"]).apply(len).unstack(0).fillna(0)
        #     ax.plot(df)
        #     ax.set_title(group)
            # ax.imshow(df, aspect=.5)
            # ax.set_yticks(list(range(len(df.index)))[::14])
            # ax.set_xticks(list(range(len(df.columns)))[::4])
            # ax.set_xticklabels(df.columns[::4], rotation=45, ha="right")
            # ax.set_yticklabels(df.index[::14])
        # plt.tight_layout()
        # plt.title("records per person per day")
        # plt.show()

    @classmethod
    def _y(cls):
        sdf = cls.df[cls.df.variable == "mood"]
        df: pd.DataFrame = sdf.groupby(["id", "date"]).value.mean().reset_index(-1)
        df["date"] -= timedelta(days=1)
        y = df.stack("date")
        return y

    @classmethod
    def data_relationships(cls):
        raise NotImplementedError(
            """
            You need to think about whether this can be shown in 1c or not - it is much easier to find relationships
            on the cleaned data using machine learning techniques than using basic plots / manually combing through
            data.
            
            The most basic is just df.cov() on the cleaned + aggregated data.
            """
        )

    @classmethod
    def main(cls):
        cls.text()
        cls.variables_distribution_of_values()
        cls.plots()
        cls.data_relationships()

    @classmethod
    def time_plot(cls):
        df = cls.df.copy()
        df = df[df.variable.str.startswith('appCat')]
        df = df[df.id == cls.df.id.unique()[0]]
        to_minute = lambda x: (t:=x.time()).hour * 100 + t.minute
        df["end"] = (df.datetime + df.value.apply(lambda x: timedelta(seconds=x))).apply(to_minute)
        df["start"] = df.datetime.apply(to_minute)
        for date, sdf in df.groupby("date"):
            sdfx = sdf.sort_values("start")
            plt.plot([x for s, e in zip(sdfx.start, sdfx.end) for x in [s, e]])
            plt.title(date)
            plt.show()


if __name__ == '__main__':
    # Task1A.time_plot()
    # Task1A.variables_distribution_of_values()
    Task1A.main()