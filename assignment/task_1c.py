from assignment.data_loader import DataLoader
from assignment.task_1b import Task1B


class Task1C:
    """
    Essentially there are two approaches you can consider to create a predictive model
    using this dataset (which we will do in the next part of this assignment):

    -> use a machine learning approach that can deal with temporal data (e.g. recurrent neural networks)

    or you can try to aggregate the history somehow to create attributes that can be used in a more common machine
    learning approach (e.g. SVM, decision tree). For instance, you use the average mood during the last five
    days as a predictor. Ample literature is present in the area of temporal data mining that describes how
    such a transformation can be made.

    For the feature engineering, you are going to focus on such a transformation in this part of the assignment.
    This is illustrated in Figure 1.
    """
    df = DataLoader.load_to_df()

    @classmethod
    def aggregate_data(cls):
        df = Task1B.remove_incorrect_values()
        print(df.variable.unique())
        # sum_agg = df.groupby(["id", "date", "variable"]).value.sum().unstack().fillna(0) / 60
        # print(sum_agg.head().to_string())


if __name__ == '__main__':
    Task1C.aggregate_data()