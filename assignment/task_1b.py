from os import remove

import pandas as pd

from assignment.data_loader import DataLoader

class Task1B:
    """
    As the insights from Task 1A will have shown, the dataset you analyze contains quite some
    noise. Values are sometimes missing, and extreme or incorrect values are seen that are likely
    outliers you may want to remove from the dataset. We will clean the dataset in two steps:


    1.  Apply an approach to remove extreme and incorrect values from your dataset. Describe
        what your approach is, why you consider that to be a good approach, and describe what
        the result of applying the approach is.


    2.  The advanced dataset contains a number of time series, select two approaches to impute
        missing values that are logical for such time series and argue for one of them based
        on the insights you gain, base yourself on insight from the data, logical reasoning
        and scientific literature. Also consider what to do with prolonged periods of missing
        data in a time series.

    Deliverables:
        -> Remove extreme and incorrect values
            -> Describe what your approach is and why you consider that to be a good approach.
            -> Describe the result of your approach

        -> select two approaches to impute missing values and argue for one of them
            -> consider what to do with prolonged periods of missing data in a time series.

    """
    df = DataLoader.load_to_df()

    def show_missing_data(self):
        record_count = self.df.groupby(["id", "date", "variable"]).apply(len).unstack().fillna(0)
        record_count.columns = [i for i, c in enumerate(record_count.columns)]
        print(record_count.to_string())

    """remove incorrect values before we aggregate data by day"""
    @staticmethod
    def remove_incorrect_values(df: pd.DataFrame) -> pd.DataFrame:
        no_invalid_data = {
            *"mood call sms circumplex.arousal circumplex.valence activity".split()
        }
        has_invalid_data = df.variable.unique()
        print(no_invalid_data.symmetric_difference(has_invalid_data))
        print(
            """
            from observing the data we found that only the 'appCat' variables had what could be described
            as extreme or incorrect values. These are time values measured in seconds - so we removed any values that
            were less than 1 second and set a value ceiling of 3 hours. 

            We consider this to be a good approach as it only removes a very small minority of the data, as opposed to
            a quantile-based approach which is guaranteed to remove a certain percentage of data regardless of whether
            it is erroneous or not. 

            - As the values are seconds, negative points should not be considered at all.
            - As we are planning on aggregating later, removing records of less than 1 second is unlikely
              to have a significant effect on our data
            - As this is data about time spent on apps, it is likely that a time exceeding 3 hours is due to someone
              accidentally leaving their phone on but is not actually using it - especially in the case of 
              non-entertainment apps such as `office` or `builtin`, and especially since the data is recorded in 2014,
              when people where less obsessed with their phones.

            From Figure !REF we can see that there are leading and trailing days where no mood data was recorded.
            Given that there was little data recorded in general on these days, we think it is sensible to remove these
            leading and trailing days from the dataset.
            """
        )
        # remove time records that are less than 0 second
        df = df[~df.variable.str.startswith("appCat") | (df.value > 0)]

        return df

    """remove extreme values in df which has unit: per day"""
    @staticmethod
    def remove_extreme_values(df: pd.DataFrame) -> pd.DataFrame:
        #clip to max 3 hours
        df[df.filter(like='total') > 3600 * 8] = 3600 * 8
        # clip to max 3 hours
        # cond = df.variable.str.startswith("appCat")
        # df.loc[cond, "value"] = df.loc[cond, "value"].clip(upper=3600 * 3)
        return df


    """
    - df unit: per day
    remove all row in which values of mood is nan
    drop all time chunks that have less than 6 data as the length of our time window is 6 (5 predict 1)
    """
    @staticmethod
    def trim_values(df: pd.DataFrame) -> pd.DataFrame:
        # nan_mask = df['average_mood'].isna()
        #
        # # Find indices where there are 3 consecutive NaNs
        # consecutive_nan_1 = nan_mask.rolling(window=5, center=True, min_periods=1).sum() > 2
        # consecutive_nan_2 = nan_mask.rolling(window=3, center=True, min_periods=1).sum() >= 2
        # consecutive_nan_3 = nan_mask.rolling(window=5, min_periods=1).sum() >= 2
        #
        # # print(consecutive_nan)
        #
        # # Filter the DataFrame
        # filtered_df = df[~consecutive_nan_1]
        # filtered_df = filtered_df[~consecutive_nan_2]
        # filtered_df = filtered_df[~consecutive_nan_3]

        df = df.dropna(subset=["average_mood"])
        df['date'] = pd.to_datetime(df['date'])
        df['date_diff'] = df['date'].diff().dt.days
        df['date_diff'].fillna(1, inplace=True)
        df['group'] = ((df['date_diff'] > 1) | (df['date_diff'] < 0)).cumsum()
        # print(df[["date", 'id', 'average_mood', 'date_diff', 'group']].to_string())
        # Filter groups with at least 6 continuous dates
        filtered_df = df.groupby('group').filter(lambda x: len(x) >= 6)
        # Drop helper columns
        filtered_df = filtered_df.drop(columns=['date_diff'])
        # print(filtered_df)
        filtered_df[["id", "date", "group", "average_mood"]].to_csv('static/df_per_day_trimmed.csv')
        filtered_df.to_csv('static/df_per_day_trimmed_features.csv')
        return filtered_df

    """
    
    """
    @staticmethod
    def set_classes(df: pd.DataFrame) -> pd.DataFrame:
        # Define the bins and labels
        bins = [0, 2, 4, 6, 7, 8, 9, 10]  # Define the range boundaries
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Define the corresponding labels

        # Create a new column for categorized moods
        df['mood_class'] = pd.cut(df['average_mood'], bins=bins, labels=labels, right=True)
        print(df["id", "date", "group", "average_mood", "mood_class"].to_string())
        return df

    def get_non_temporal_data(self, df: pd.DataFrame):
        trimmed_data = self.trim_values(df)

        result = []
        for group, sdf in list(trimmed_data.groupby("group")):
            sdf['last_5day_avg_mood'] = sdf.shift()["average_mood"].rolling(5).mean()
            result.append(sdf)
        final_df = pd.concat(result, ignore_index=True)
        final_df[["id", "date", "group", "average_mood", "last_5day_avg_mood"]].to_csv('static/df_non_temporal.csv')

    def get_temporal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def get_values_per_day(self):
        """
        this also implicitly imputes missing values for variables except mood, arousal, valence and activity:
        if the variable has no value/ no records on that day, we impute it values to 0
        """
        df = self.df.copy()
        df = self.remove_incorrect_values(df)
        print(df)
        print("==="*10)
        df = df[["id", "date", "variable", "value"]]

        result = (
            df.groupby(['id', 'date'])
            .agg(
                average_mood=('value', lambda x: x[df['variable'] == 'mood'].mean()),
                average_arousal=('value', lambda x: x[df['variable'] == 'circumplex.arousal'].mean()),
                average_valence=('value', lambda x: x[df['variable'] == 'circumplex.valence'].mean()),
                average_activity=('value', lambda x: x[df['variable'] == 'activity'].mean()),
                total_screen=('value', lambda x: x[df['variable'] == 'screen'].sum()),
                total_sms=('value', lambda x: x[df['variable'] == 'sms'].sum()),
                total_calls=('value', lambda x: x[df['variable'] == 'call'].sum()),
                total_built_in=('value', lambda x: x[df['variable'] == 'appCat.builtin'].sum()),
                total_communication=('value', lambda x: x[df['variable'] == 'appCat.communication'].sum()),
                total_entertainment=('value', lambda x: x[df['variable'] == 'appCat.entertainment'].sum()),
                total_finance=('value', lambda x: x[df['variable'] == 'appCat.finance'].sum()),
                total_game=('value', lambda x: x[df['variable'] == 'appCat.game'].sum()),
                total_office=('value', lambda x: x[df['variable'] == 'appCat.office'].sum()),
                total_other=('value', lambda x: x[df['variable'] == 'appCat.other'].sum()),
                total_social=('value', lambda x: x[df['variable'] == 'appCat.social'].sum()),
                total_travel=('value', lambda x: x[df['variable'] == 'appCat.travel'].sum()),
                total_unknown=('value', lambda x: x[df['variable'] == 'appCat.unknown'].sum()),
                total_utilities=('value', lambda x: x[df['variable'] == 'appCat.utilities'].sum()),
                total_weather=('value', lambda x: x[df['variable'] == 'appCat.weather'].sum()),
            ).reset_index()
        )
        print("===" * 10)
        result = self.remove_extreme_values(result)
        result.to_csv('static/df_per_day.csv')
        print(result.to_string())


if __name__ == '__main__':
    df = DataLoader.load_to_df()
    task1B = Task1B()
    # task1B.get_values_per_day()
    data_per_day = pd.read_csv('static/df_per_day.csv').drop(columns="Unnamed: 0")
    task1B.get_non_temporal_data(data_per_day)
    task1B.get_temporal_data(data_per_day)
    # df = Task1B.remove_incorrect_values
    # Task1B.get_values_per_day()
    # df_per_day = pd.read_csv('static/df_per_day.csv').drop(columns="Unnamed: 0")
    # df_per_day_no_extreme = Task1B.remove_extreme_values(df_per_day)
    # Task1B.trim_values_by_mood(df_per_day_no_extreme)
    # df_temporal_data = Task1B.get_temporal_data()
    # df_non_temporal_data = Task1B.get_non_temporal_data()

