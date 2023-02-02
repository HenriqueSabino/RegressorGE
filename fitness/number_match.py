from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff


class number_match(base_ff):
    """Fitness function for matching a number. Takes a number and returns
    fitness. Penalises output that is not the same number as the target."""

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # Set target string.
        self.target = params['TARGET']

    def evaluate(self, ind, **kwargs):
        expression = ind.phenotype
        expression_parts = expression.split(" ")

        result = int(expression_parts[0])

        for i in range(1, len(expression_parts), 2):
            if expression_parts[i] == "+":
                result += int(expression_parts[i + 1])
            else:
                result -= int(expression_parts[i + 1])

        return abs(result - self.target)
