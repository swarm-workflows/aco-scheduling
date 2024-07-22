""" Define MixVars class for mealpy

"""

from mealpy import GA
from mealpy.utils.space import BaseVar, PermutationVar, LabelEncoder
import numpy as np


class MixVars(BaseVar):
    r"""Mixed Permutation Variables."""

    def __init__(self, vars=[1], name="mix_var"):
        r""" Initialize the class

        Args:
            vars (list): list of number of variables for each type of variables
            name (str): name of the variables
        """
        #
        self.vars = vars
        self.len = len(vars)
        self.set_n_vars(sum(vars))
        raise NotImplementedError("This class is not implemented yet.")

    def encode(self, x):
        raise NotImplementedError("This class is not implemented yet.")

    def decode(self, x):

        raise NotImplementedError("This class is not implemented yet.")

    def correct(self, x):
        r"""Validate the variable given bounds.
        """

        raise NotImplementedError("This class is not implemented yet.")

    def generate(self):
        r"""Generate variables
        """
        Xs = []
        for _var in self.vars:
            _x = self.generator.permutation(_var)
            Xs.append(_x)
        return np.concatenate(Xs)


class MixPermVars(PermutationVar):
    r""" Mixture of permutation variables. """

    def __init__(self, input_sizes=[1], name="mix_perm_var"):
        r""" Given a list of input sizes, this class will generate a list of permutation variables.

        Args:
            input_sizes (list): list of number of variables for each type of variables
            name (str): name of the variables
        """
        #
        self.name = name
        self.input_sizes = input_sizes
        self.len = len(self.input_sizes)
        self.set_n_vars(sum(self.input_sizes))
        self.perm_vars = []
        for i, var in enumerate(self.input_sizes):
            _perm_var = PermutationVar(valid_set=list(range(0, var)), name=f"var_{i}")
            self.perm_vars.append(_perm_var)
        self.eps = 1e-4
        self.le = LabelEncoder().fit(np.concatenate([np.arange(var) for var in self.input_sizes]))
        self.lb = np.zeros(self.n_vars)
        self.ub = np.concatenate([np.ones(var) * (var - self.eps) for var in self.input_sizes])

    def encode(self, x):
        r""" Encode the variables

        Args:
            x (np.array): input variables

        Returns:
            np.array: encoded variables
        """
        return np.array(x, dtype=float)

    def decode(self, x):
        r""" Decode the variables to original variables.

        Args:
            x (np.array): input variables

        Returns:
            np.array: decoded variables
        """
        decoded_vars = []
        start_index = 0
        for var in self.input_sizes:
            end_index = start_index + var
            slice_x = x[start_index:end_index]
            decoded_slice = super().decode(slice_x)
            decoded_vars.append(decoded_slice)
            start_index = end_index
        return np.concatenate(decoded_vars)

    def correct(self, x):
        r""" Correct the variables to valid bounds.

        Args:
            x (np.array): input variables

        Returns:
            np.array: corrected variables
        """
        corrected_vars = []
        start_index = 0
        for var in self.input_sizes:
            end_index = start_index + var
            slice_x = x[start_index:end_index]
            corrected_slice = super().correct(slice_x)
            corrected_vars.append(corrected_slice)
            start_index = end_index
        return np.concatenate(corrected_vars)

    def generate(self):
        r"""Generate variables"""
        Xs = []
        for _var in self.perm_vars:
            _x = _var.generate()
            Xs.append(_x)
        return np.concatenate(Xs)


if __name__ == "__main__":
    def func(x):
        """ test function
        f(x) = x[0]^2 + x[3]^2 + x[4]^2 + x[5]^2 + x[8]^2
        """
        return np.sum(x[0]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[8]**2)

    prob = {
        "obj_func": func,
        "bounds": MixPermVars(input_sizes=[2, 3, 4]),
        "minmax": "min",
    }

    optimizer = GA.BaseGA(epoch=100, pop_size=10, pc=0.85, pm=0.1)
    optimizer.solve(prob)

    print(optimizer.g_best.solution)
    print(optimizer.g_best.target.fitness)
