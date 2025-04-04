from statistics import LinearRegression

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

    @classmethod
    def show_missing_data(cls):
        record_count = cls.df.groupby(["id", "date", "variable"]).apply(len).unstack().fillna(0)
        record_count.columns = [i for i, c in enumerate(record_count.columns)]
        print(record_count.to_string())

    @classmethod
    def remove_incorrect_values(cls):
        no_invalid_data = {
            *"mood call sms circumplex.arousal circumplex.valence activity".split()
        }
        has_invalid_data = cls.df.variable.unique()
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
        df = cls.df.copy()

        # remove time records that are less than 1 second
        df = df[~(df.variable.str.startswith("appCat") & df.value < 1)]
        cond = df.variable.str.startswith("appCat")
        df.loc[cond, "value"] = df.loc[cond, "value"].clip(upper=3600 * 3) # clip to max 3 hours

    @classmethod
    def linear_regr(cls):
        df = cls.df.groupby(["date", "variable"]).apply(len).unstack().fillna(0)
        print(df.iloc[20:30].to_string())
        y = df["mood"]
        # X = df[[col for col in df.columns if col != "mood"]]
        X = df.drop(columns="mood")
        X = df[["screen"]]
        from sklearn.linear_model import LinearRegression, LogisticRegression
        model = LinearRegression().fit(X, y)
        print(model.score(X, y))
        # print(model.coef_)



# Task1B.remove_incorrect_values()
Task1B.linear_regr()