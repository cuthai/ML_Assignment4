import numpy as np


class LogisticRegressor:
    """
    Class LogisticRegressor
    """
    def __init__(self, etl):
        """
        Init function

        Sets main variables and array conversion

        :param etl: etl, etl object with transformed and split data
        """
        # Meta Variables
        self.etl = etl
        self.data_name = self.etl.data_name
        self.class_names = etl.class_names
        self.classes = etl.classes

        # Tune Variables
        self.step_size = .01

        # Data Variables
        self.tune_data = etl.tune_data
        self.test_split = etl.test_split
        self.train_split = etl.train_split

        # Array Variables
        self.tune_array = self.tune_data.to_numpy()
        self.test_array_split = {key: self.test_split[key].to_numpy() for key in self.test_split.keys()}
        self.train_array_split = {key: self.train_split[key].to_numpy() for key in self.train_split.keys()}

        # Train Models
        self.train_models = {}

        # Tune Results
        self.tune_results = {}

        # Test Results
        self.test_results = {}

        # Summary
        self.summary = {}
        self.summary_classification = None

    def fit(self):
        for index in range(5):
            self.train(self.train_array_split[index])

    def train(self, train_data):
        i = 1  # remove

        train_x = train_data[:, :-1].astype(float)
        train_y = train_data[:, -1]

        weights = np.random.uniform(low=-.01, high=.01, size=(self.classes, train_x.shape[1]))

        misclassification = 1

        while True:
            weights_delta = np.ndarray((self.classes, train_x.shape[1]))
            outputs = np.matmul(train_x, weights.T)
            likelihood = np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:, None]
            predictions = np.argmax(likelihood, axis=1).astype('O')

            for index in range(self.classes):
                current_class = self.class_names[index]
                actuals = (train_y == current_class).astype(int)

                weights_delta[index, :] = np.matmul((actuals - likelihood[:, index]), train_x)

                predictions[predictions == index] = self.class_names[index]

            weights = weights + (self.step_size * weights_delta)

            new_misclassification = sum(predictions != train_y) / len(train_y)

            if new_misclassification < misclassification:
                misclassification = new_misclassification
                print('learning cycle: ', i)
                print(misclassification)
                i += 1
            else:
                print('stop')
                print(misclassification)
                break

        pass
