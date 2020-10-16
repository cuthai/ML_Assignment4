import numpy as np
import pandas as pd


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
        self.train_models = {index: {} for index in range(5)}

        # Tune Results
        self.tune_results = {
            round(step_size, 2): {} for step_size in np.linspace(.01, .25, 25)
        }

        # Test Results
        self.test_results = {index: {} for index in range(5)}

        # Summary
        self.summary = {}
        self.summary_classification = None

    def tune(self):
        step_sizes = self.tune_results.keys()

        for step_size in step_sizes:
            misclassification = 0

            for index in range(5):
                data = self.train_array_split[index]
                weights, intercepts = self.train(data, step_size)

                model = {
                    'weights': weights,
                    'intercepts': intercepts
                }
                predictions = self.classify(self.tune_array, model)

                results = pd.DataFrame.copy(self.tune_data)
                results['Prediction'] = predictions

                misclassification += len(results[results['Class'] != results['Prediction']]) / len(results)

            self.tune_results[step_size].update({
                'misclassification': misclassification / 5
            })

    def fit(self):
        for index in range(5):
            data = self.train_array_split[index]

            weights, intercepts = self.train(data)

            self.train_models[index].update({
                'weights': weights,
                'intercepts': intercepts
            })

    def train(self, data, step_size=None):
        if not step_size:
            step_size = self.step_size

        i = 1  # remove

        x = data[:, :-1].astype(float)
        y = data[:, -1]

        weights = np.random.uniform(low=-.01, high=.01, size=(self.classes, x.shape[1]))
        intercepts = np.zeros((self.classes, 1))

        misclassification = 1

        while True:
            weights_delta = np.zeros((self.classes, x.shape[1]))
            intercepts_delta = np.zeros((self.classes, 1))

            outputs = np.matmul(x, weights.T) + intercepts.T
            likelihood = (np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:, None])
            predictions = np.argmax(likelihood, axis=1).astype('O')

            for index in range(self.classes):
                current_class = self.class_names[index]
                actuals = (y == current_class).astype(int)
                difference = (actuals - likelihood[:, index])

                weights_delta[index, :] = np.matmul(difference, x)
                intercepts_delta[index, :] += sum(difference)

                predictions[predictions == index] = self.class_names[index]

            weights = weights + (step_size * weights_delta)
            intercepts = intercepts + (step_size * intercepts_delta)

            new_misclassification = sum(predictions != y) / len(y)

            if new_misclassification < misclassification:
                misclassification = new_misclassification
                print('learning cycle: ', i)  # remove
                print(misclassification)  # remove
                i += 1  # remove
            else:
                print('stop')  # remove
                print(misclassification)  # remove
                break

        return weights, intercepts

    def predict(self):
        for index in range(5):
            data = self.test_array_split[index]
            model = self.train_models[index]

            predictions = self.classify(data, model)

            results = pd.DataFrame.copy(self.test_split[index])
            results['Prediction'] = predictions

            misclassification = len(results[results['Class'] != results['Prediction']]) / len(results)

            self.test_results[index].update({
                'results': results,
                'misclassification': misclassification
            })

    def classify(self, data, model):
        x = data[:, :-1].astype(float)

        weights = model['weights']
        intercepts = model['intercepts']

        outputs = np.matmul(x, weights.T) + intercepts.T
        likelihood = (np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:, None])
        predictions = np.argmax(likelihood, axis=1).astype('O')

        for index in range(self.classes):
            predictions[predictions == index] = self.class_names[index]

        return predictions
