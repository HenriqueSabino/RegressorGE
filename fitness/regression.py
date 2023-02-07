from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
import requests
from sewar.full_ref import mse
from openpyxl import load_workbook
import numpy as np

class regression(base_ff):

    maximise = False
    multi_objective = True

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.num_obj = 2
        fit = base_ff()
        fit.maximise = False
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

    def load_data(self):
        wb = load_workbook(params['DATASET_NAME'])
        dataset_name = ["REDWINE","SUNSPOT","B1H","POLLUTION","GAS","LAKEERIE","Electricity","PIGS","Nordic","CARSALES"]
        dataset = {}
        for name in dataset_name:

            ws = wb[name]
            columns = [ws["A"],ws["B"],ws["C"],ws["D"],ws["E"],ws["F"],ws["G"]]

            sheet = [[] for _ in range(len(columns[0]))]

            for c in columns:
                for index,item in enumerate(c):
                    sheet[index].append(item.value)

            sheet = np.array(sheet)
            dataset[name] = {"raw":sheet}

        for name,data in dataset.items():
            sheet = dataset[name]["raw"]

            dataset[name]["series"] = {"treino":[],"teste":[]}
            dataset[name]["arima"] = {"treino":[],"teste":[]}
            dataset[name]["mlp"] = {"treino":[],"teste":[]}
            dataset[name]["srv"] = {"treino":[],"teste":[]}
            dataset[name]["rbf"] = {"treino":[],"teste":[]}

            for row in sheet[1:]:
                split_type = row[2]
                if row[4]==None:
                    continue
                key=None
                if split_type == "Teste":
                    key="teste"
                elif split_type == "Validacao":
                    key="treino"
                else:
                    continue

                dataset[name]["series"][key].append(float(row[1]))
                dataset[name]["arima"][key].append(float(row[3]))
                dataset[name]["mlp"][key].append(float(row[4]))
                dataset[name]["srv"][key].append(float(row[5]))
                dataset[name]["rbf"][key].append(float(row[6]))

            dataset[name]["series"]["treino"]=np.array(dataset[name]["series"]["treino"])
            dataset[name]["arima"]["treino"]=np.array(dataset[name]["arima"]["treino"])
            dataset[name]["mlp"]["treino"]=np.array(dataset[name]["mlp"]["treino"])
            dataset[name]["srv"]["treino"]=np.array(dataset[name]["srv"]["treino"])
            dataset[name]["rbf"]["treino"]=np.array(dataset[name]["rbf"]["treino"])


            dataset[name]["series"]["teste"]=np.array(dataset[name]["series"]["teste"])
            dataset[name]["arima"]["teste"]=np.array(dataset[name]["arima"]["teste"])
            dataset[name]["mlp"]["teste"]=np.array(dataset[name]["mlp"]["teste"])
            dataset[name]["srv"]["teste"]=np.array(dataset[name]["srv"]["teste"])
            dataset[name]["rbf"]["teste"]=np.array(dataset[name]["rbf"]["teste"])

        self.dataset = dataset["REDWINE"]

    def build_model(self, phenotype):
        self.load_data()
        model = {"linear":{},"nlinear":{}}
        pesos = 1

        print(phenotype)

        model_parts = phenotype.split(';')
        linear = model_parts[0]
        nlinear = model_parts[1]

        model["linear"][linear] = pesos

        nlinear_parts = nlinear.split(',')

        for i in range(len(nlinear_parts)):
            model["nlinear"][nlinear_parts[i]] = pesos

        return model

    def train_model(self, model):
        pass

    def evaluate(self, ind, **kwargs):
        #self.load_data()
        # accuracy, accuracy_sd, f1_score, f1_score_sd = self.get_metrics(ind.phenotype)

        # if accuracy is None and f1_score is None:
        model = self.build_model(ind.phenotype)
        print(model)
        predict = np.zeros(len(self.dataset["arima"]["teste"]))

        for key,val in model["linear"].items():
            predict += self.dataset[key]["teste"] * val
        
        for key,val in model["nlinear"].items():

            predict += self.dataset[key]["teste"] * val

        
        return mse(self.dataset['series']['teste'],predict) , 0


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
