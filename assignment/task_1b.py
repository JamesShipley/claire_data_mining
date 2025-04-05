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
            were less than 1 second or more than 3 hours 
            """
        )


Task1B.remove_incorrect_values()