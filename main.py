from utils.args import args
from etl.etl import ETL


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

    # Decision Tree
    # Classification
    if arguments.data_name in ['breast-cancer', 'car', 'segmentation']:
        # Set up kwargs
        kwargs = {
            'etl': etl,
            'prune': arguments.prune
        }

        dt_model = None
    # Regression
    else:
        # Set up kwargs
        kwargs = {
            'etl': etl,
            'percent_threshold': arguments.percent_threshold
        }

        dt_model = None

        # Tune
        if arguments.tune:
            dt_model.tune()

    # Fit
    dt_model.fit()

    # Predict
    dt_model.predict()

    # Summarize
    dt_model.summarize()


if __name__ == '__main__':
    main()
