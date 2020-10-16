from utils.args import args
from etl.etl import ETL
from logistic_regression.logistic_regression import LogisticRegressor


def main():
    """
    Main function to run Decision Tree Classifier/Regressor
    """
    # Parse arguments
    arguments = args()

    # Set up kwargs for ETL
    kwargs = {
        'data_name': arguments.data_name,
        'random_state': arguments.random_state
    }
    etl = ETL(**kwargs)

    # Set up kwargs and create object
    kwargs = {
        'etl': etl
    }
    model = LogisticRegressor(**kwargs)

    # Fit
    model.fit()

    # Predict
    model.predict()

    # Summarize
    # model.summarize()

    pass


if __name__ == '__main__':
    main()
