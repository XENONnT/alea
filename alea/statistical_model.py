import inspect
from inference_interface import toydata_from_file, toydata_to_file
from typing import Tuple, Optional
import numpy as np
from iminuit import Minuit
from iminuit.util import make_func_code
from alea.parameters import Parameters

class StatisticalModel:
    """
    Class that defines a statistical model.
    The statisical model contains two parts that you must define yourself:
        - a likelihood function, ll(self, parameter_1, parameter_2... parameter_n):
            a function of a set of named parameters
            returns a float expressing the loglikelihood for observed data
            given these parameters
        - a data generation method generate_data(self, parameter_1, parameter_2... parameter_n):
            a function of the same set of named parameters
            returns a full data set:

    Methods:
         __init__
         required to implement:

         _ll
         _generate_data

         optional to implement:
         get_mus
         get_likelihood_term_names

         Implemented here:
         store_data
         fit
         get_confidence_interval
         get_expectation_values
         get_parameter_list
         print_config


    Other members:
        _data = None
        _config = {}
        _confidence_level = 0.9
        _confidence_interval_kind = "upper,lower,unified"
        _fit_guess = {}
        _fixed_parameters = []
    """
    def ll(self, **kwargs) -> float:
        # CAUTION: This implementation won't allow you to call the likelihood by positional arguments.
        parameters = self.parameters(**kwargs)
        return self._ll(**parameters)

    def _ll(self, **kwargs) -> float:
        raise NotImplementedError("You must write a likelihood function (_ll) for your statistical model or use a subclass where it is written for you")

    def generate_data(self, **kwargs):
        # CAUTION: This implementation won't allow you to call generate_data by positional arguments.
        if not self.parameters.values_in_fit_limits(**kwargs):
            raise ValueError("Values are not within fit limits")
        generate_values = self.parameters(**kwargs)
        return self._generate_data(**generate_values)

    def _generate_data(self, **kwargs):
        raise NotImplementedError("You must write a data-generation method (_generate_data) for your statistical model or use a subclass where it is written for you")

    def __init__(self,
                 data = None,
                 parameter_definition: dict or list = None,
                 confidence_level: float = 0.9,
                 confidence_interval_kind:str = "unified",
                 **kwargs):
        self._data = data
        self._confidence_level = confidence_level
        self._confidence_interval_kind = confidence_interval_kind
        self._define_parameters(parameter_definition)

        self._check_ll_and_generate_data_signature()

    @property
    def data(self):
        """
        Simple getter for a data-set-- mainly here so it can be over-ridden for special needs.
        Data-sets are expected to be in the form of a list of one or more structured arrays-- representing the data-sets of one or more likelihood terms.
        """
        if self._data is None:
            raise Exception("data has not been assigned this statistical model!")
        return self._data

    @data.setter
    def data(self, data):
        """
        Simple setter for a data-set-- mainly here so it can be over-ridden for special needs.
        Data-sets are expected to be in the form of a list of one or more structured arrays-- representing the data-sets of one or more likelihood terms.
        """
        self._data = data

    def store_data(self, file_name, data_list, data_name_list=None, metadata = None):
        """
        Store a list of datasets (each on the form of a list of one or more structured arrays)
        Using inference_interface, but included here to allow over-writing.
        structure would be: [[datasets1],[datasets2]... [datasetsn]]
        where each of datasets is a list of structured arrays
        if you specify, it is set, if not it will read from self.get_likelihood_term_names
        if not defined, it will be ["0","1"..."n-1"]
        """
        if data_name_list is None:
            try:
                data_name_list = self.get_likelihood_term_names()
            except NotImplementedError:
                data_name_list = ["{:d}".format(i) for i in range(len(data_list[0]))]

        kw = dict(metadata = metadata) if metadata is not None else dict()
        toydata_to_file(file_name, data_list, data_name_list, **kw)



    def get_confidence_interval(self) -> Tuple[float, float]:
        return NotImplementedError("todo")
    def get_expectation_values(self):
        return NotImplementedError("get_expectation_values is optional to implement")

    def get_likelihood_term_from_name(self, likelihood_name):
        """
        returns the index of a likelihood term if the likelihood has several names
        """
        if hasattr(self, "likelihood_names"):
            likelihood_names = self.likelihood_names
            return {n:i for i,n in enumerate(likelihood_names)}[likelihood_name]
        else:
            raise NotImplementedError("The attribute likelihood_names is not defined.")


    def get_parameter_list(self):
        """returns a set of all parameters that the generate_data and likelihood accepts"""
        return self._parameter_list

    def print_config(self):
        for k,i in self.config:
            print(k,i)

    def make_objective(self, minus=True, **kwargs):
        sign = -1 if minus else 1

        def cost(args):
            # Get the arguments from args, then fill in the ones already fixed in outer kwargs
            call_kwargs = {}
            for i, k in enumerate(self.parameters.names):
                call_kwargs[k] = args[i]
            # call_kwargs.update(kwargs)
            return self.ll(**call_kwargs) * sign

        return cost

    def fit(self, verbose=False, **kwargs):
        fixed_parameters = list(kwargs.keys())
        guesses = self.parameters.fit_guesses
        guesses.update(kwargs)
        if not self.parameters.values_in_fit_limits(**guesses):
            raise ValueError("Initial guesses are not within fit limits")
        defaults = self.parameters(**guesses)

        cost = self.make_objective(minus=True, **kwargs)

        class MinuitWrap:
            """Wrapper for functions to be called by Minuit

            s_args must be a list of argument names of function f
            the names in this list must be the same as the keys of
            the dictionary passed to the Minuit call."""

            def __init__(self, f, s_args):
                self.func = f
                self.s_args = s_args
                self.func_code = make_func_code(s_args)

            def __call__(self, *args):
                return self.func(args)

        # Make the Minuit object
        cost.errordef = Minuit.LIKELIHOOD
        m = Minuit(MinuitWrap(cost, s_args=self.parameters.names),
                   **defaults)
        fixed_params = [] if fixed_parameters is None else fixed_parameters
        fixed_params += self.parameters.not_fittable
        for par in fixed_params:
            m.fixed[par] = True
        for n, l in self.parameters.fit_limits.items():
            m.limits[n] = l

        # Call migrad to do the actual minimization
        m.migrad()
        if verbose:
            print(m)
        return m.values.to_dict(), -1 * m.fval

    def _check_ll_and_generate_data_signature(self):
        ll_params = set(inspect.signature(self._ll).parameters)
        generate_data_params = set(inspect.signature(self._generate_data).parameters)
        if ll_params != generate_data_params:
            raise AssertionError("ll and generate_data must have the same signature (parameters)")

    def _define_parameters(self, parameter_definition):
        if parameter_definition is None:
            self.parameters = Parameters()
        elif isinstance(parameter_definition, dict):
            self.parameters = Parameters.from_config(parameter_definition)
        elif isinstance(parameter_definition, list):
            self.parameters = Parameters.from_list(parameter_definition)
        else:
            raise Exception("parameter_definition must be dict or list")
