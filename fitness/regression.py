from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
import requests


class regression(base_ff):

    maximise = True
    multi_objective = True

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.num_obj = 2
        fit = base_ff()
        fit.maximise = True
        self.fitness_functions = [fit, fit]
        self.default_fitness = [float('nan'), float('nan')]

    def get_metrics(self, phenotype):

        accuracy, accuracy_sd, f1_score, f1_score_sd = None, None, None, None

        r = requests.get(params['METRICS_URL'], params={
            'dataset': params['DATASET_NAME'],
            'phenotype': phenotype,
        })
        data = r.json()

        if len(data):
            data = data[0]
            accuracy = float(data['accuracy'])
            accuracy_sd = float(data['accuracy_sd'])
            f1_score = float(data['f1_score'])
            f1_score_sd = float(data['f1_score_sd'])

        return accuracy, accuracy_sd, f1_score, f1_score_sd

    def save_metrics(self, phenotype, accuracy, accuracy_sd, f1_score, f1_score_sd):
        data = {
            'dataset': params['DATASET_NAME'],
            'phenotype': phenotype,
            'accuracy': accuracy,
            'accuracy_sd': accuracy_sd,
            'f1_score': f1_score,
            'f1_score_sd': f1_score_sd,
        }
        requests.post(params['METRICS_URL'], json=data)

    def build_model(self, phenotype):

        print(phenotype)

        model_parts = phenotype.split(';')
        linear = model_parts[0]
        nlinear = model_parts[1]

        fitness = [0, 1]

        if linear == 'arima':
            fitness[0] += 1
            fitness[1] += 1
        elif linear == 'sarima':
            fitness[0] += 2
            fitness[1] += 2
        elif linear == 'ets':
            fitness[0] += 3
            fitness[1] += 3

        nliear_parts = nlinear.split(',')

        for i in range(len(nliear_parts)):
            if nliear_parts[i] == 'svm':
                fitness[0] += 1
                fitness[1] += 1
            elif nliear_parts[i] == 'mlp':
                fitness[0] += 2
                fitness[1] += 2
            elif nliear_parts[i] == 'lstm':
                fitness[0] += 3
                fitness[1] += 3
            elif nliear_parts[i] == 'cnn':
                fitness[0] += 4
                fitness[1] += 4

        return fitness

    def train_model(self, model):
        pass

    def evaluate(self, ind, **kwargs):

        # accuracy, accuracy_sd, f1_score, f1_score_sd = self.get_metrics(ind.phenotype)

        # if accuracy is None and f1_score is None:
        model = self.build_model(ind.phenotype)
        # accuracy, accuracy_sd, f1_score, f1_score_sd = self.train_model(model)
        # self.save_metrics(ind.phenotype, accuracy, accuracy_sd, f1_score, f1_score_sd)

        return model

    @staticmethod
    def value(fitness_vector, objective_index):
        """
        This is a static method required by NSGA-II for sorting populations
        based on a given fitness function, or for returning a given index of a
        population based on a given fitness function.
        :param fitness_vector: A vector/list of fitnesses.
        :param objective_index: The index of the desired fitness.
        :return: The fitness at the objective index of the fitness vecror.
        """

        if not isinstance(fitness_vector, list):
            return float("inf")

        return fitness_vector[objective_index]
