import inspect
import warnings
from typing import Tuple, Callable, Optional

import numpy as np
from scipy.stats import chi2
from scipy.optimize import brentq
from iminuit import Minuit
from blueice.likelihood import _needs_data
from inference_interface import toydata_to_file

from alea.parameters import Parameters


class StatisticalModel:
    """
    Class that defines a statistical model.

    - The statisical model contains two parts that you must define yourself:
        - a likelihood function
            ll(self, parameter_1, parameter_2... parameter_n):
            A function of a set of named parameters which
            returns a float expressing the loglikelihood for observed data given these parameters.
        - a data generation function
            generate_data(self, parameter_1, parameter_2... parameter_n)
            A function of the same set of named parameters returns a full data set.
    - Methods that you must implement:
        - _ll
        - _generate_data
    - Methods that you may implement:
        - get_expectation_values
    - Methods that already exist here:
        - ll
        - store_data
        - fit
        - get_parameter_list

    :ivar data: data of the model
    :ivar _data: data of the model
    :ivar _confidence_level: confidence level for confidence intervals
    :ivar _confidence_interval_kind: kind of confidence interval to compute
    :ivar parameters: parameters of the model
    :ivar confidence_interval_threshold: threshold for confidence interval
    :ivar is_data_set: True if data is set
    :vartype is_data_set: bool

    :param data: pre-set data of the model
    :param parameter_definition: definition of the parameters of the model
    :type parameter_definition: dict or list
    :param confidence_level: confidence level for confidence intervals
    :type confidence_level: float
    :param confidence_interval_kind: kind of confidence interval to compute
    :type confidence_interval_kind: str
    :param confidence_interval_threshold: threshold for confidence interval
    :type confidence_interval_threshold: Callable[[float], float]
    """

    def __init__(
            self,
            data = None,
            parameter_definition: Optional[dict or list] = None,
            confidence_level: float = 0.9,
            confidence_interval_kind: str = "central",  # one of central, upper, lower
            confidence_interval_threshold: Callable[[float], float] = None,
        ):
        """Initialize a statistical model"""
        if type(self) == StatisticalModel:
            raise RuntimeError(
                "You cannot instantiate the StatisticalModel class directly, "
                "you must use a subclass where the likelihood function and data generation "
                "method are implemented")

        # following https://github.com/JelleAalbers/blueice/blob/
        # 7c10222a13227e78dc7224b1a7e56ff91e4a8043/blueice/likelihood.py#L97
        self.is_data_set = False
        if data is not None:
            self.data = data
        self._confidence_level = confidence_level
        if confidence_interval_kind not in {"central", "upper", "lower"}:
            raise ValueError("confidence_interval_kind must be one of central, upper, lower")
        self._confidence_interval_kind = confidence_interval_kind
        self.confidence_interval_threshold = confidence_interval_threshold
        self._define_parameters(parameter_definition)

        self._check_ll_and_generate_data_signature()

    def _define_parameters(self, parameter_definition):
        """Initialize the parameters of the model"""
        if parameter_definition is None:
            self.parameters = Parameters()
        elif isinstance(parameter_definition, dict):
            self.parameters = Parameters.from_config(parameter_definition)
        elif isinstance(parameter_definition, list):
            self.parameters = Parameters.from_list(parameter_definition)
        else:
            raise RuntimeError("parameter_definition must be dict or list")

    def _check_ll_and_generate_data_signature(self):
        """Check that the likelihood and generate_data functions have the same signature"""
        ll_params = set(inspect.signature(self._ll).parameters)
        generate_data_params = set(inspect.signature(self._generate_data).parameters)
        if ll_params != generate_data_params:
            raise AssertionError(
                "ll and generate_data must have the same signature (parameters)")

    def _ll(self, **kwargs) -> float:
        """Likelihood function, returns the loglikelihood for the given parameters."""
        raise NotImplementedError(
            "You must write a likelihood function (_ll) for your statistical model"
            " or use a subclass where it is written for you")

    def _generate_data(self, **kwargs):
        """Generate data for the given parameters."""
        raise NotImplementedError(
            "You must write a data-generation method (_generate_data) for your statistical model"
            " or use a subclass where it is written for you")

    @_needs_data
    def ll(self, **kwargs) -> float:
        """
        Likelihod function, returns the loglikelihood for the given parameters.
        The parameters are passed as keyword arguments, positional arguments are not possible.
        If a parameter is not given, the default value is used.

        :param kwargs: keyword arguments for the parameters
        :return: likelihood value
        :rtype: float
        """
        parameters = self.parameters(**kwargs)
        return self._ll(**parameters)

    def generate_data(self, **kwargs):
        """
        Generate data for the given parameters.
        The parameters are passed as keyword arguments, positional arguments are not possible.
        If a parameter is not given, the default values are used.

        :raises ValueError: If the parameters are not within the fit limits
        :return: generated data
        """
        if not self.parameters.values_in_fit_limits(**kwargs):
            raise ValueError("Values are not within fit limits")
        generate_values = self.parameters(**kwargs)
        return self._generate_data(**generate_values)

    @property
    def data(self):
        """
        Simple getter for a data-set-- mainly here so it can be over-ridden for special needs.
        Data-sets are expected to be in the form of a list of one or more structured arrays,
        representing the data-sets of one or more likelihood terms.
        """
        if self._data is None:
            raise RuntimeError("data has not been assigned this statistical model!")
        return self._data

    @data.setter
    def data(self, data):
        """data setter"""
        self._data = data
        self.is_data_set = True

    def store_data(
            self,
            file_name, data_list, data_name_list=None, metadata=None):
        """
        Store a list of datasets (each on the form of a list of one or more structured arrays)
        Using inference_interface, but included here to allow over-writing.
        The structure would be: [[datasets1], [datasets2], ..., [datasetsn]],
        where each of datasets is a list of structured arrays.
        If you specify, it is set, if not it will read from self.get_likelihood_term_names.
        If not defined, it will be ["0", "1", ..., "n-1"]. The metadata is optional.

        :param file_name: name of the file to store the data in
        :type file_name: str
        :param data_list: list of datasets
        :type data_list: list
        :param data_name_list: list of names of the datasets
        :type data_name_list: list
        :param metadata: metadata to store with the data
        :type metadata: dict
        """
        if data_name_list is None:
            if hasattr(self, "likelihood_names"):
                data_name_list = self.likelihood_names
            else:
                data_name_list = ["{:d}".format(i) for i in range(len(data_list[0]))]

        kw = {'metadata': metadata} if metadata is not None else dict()
        toydata_to_file(file_name, data_list, data_name_list, **kw)

    def get_expectation_values(self, **parameter_values):
        """
        Get the expectation values for the of the measurement.

        :param parameter_values: values of the parameters
        """
        return NotImplementedError("get_expectation_values is optional to implement")

    @property
    def nominal_expectation_values(self):
        """
        Nominal expectation values for the sources of the likelihood.

        For this to work, you must implement `get_expectation_values`.
        """
        # no kwargs for nominal
        return self.get_expectation_values()

    def get_likelihood_term_from_name(self, likelihood_name: str):
        """
        Returns the index of a likelihood term if the likelihood has several names

        :param likelihood_name: name of the likelihood term
        :type likelihood_name: str
        :return: index of the likelihood term
        :rtype: dict
        """
        if hasattr(self, "likelihood_names"):
            likelihood_names = self.likelihood_names
            return {n: i for i, n in enumerate(likelihood_names)}[likelihood_name]
        else:
            raise NotImplementedError("The attribute likelihood_names is not defined.")

    def get_parameter_list(self):
        """Returns a set of all parameters that the generate_data and likelihood accepts"""
        return self.parameters.names

    def make_objective(self, minus=True, **kwargs):
        """
        Make a function that can be passed to Minuit

        :param minus: if True, the function is multiplied by -1
        :type minus: bool
        :return: function that can be passed to Minuit
        :rtype: Callable
        """
        sign = -1 if minus else 1

        def cost(args):
            # Get the arguments from args,
            # then fill in the ones already fixed in outer kwargs
            call_kwargs = {}
            for i, k in enumerate(self.parameters.names):
                call_kwargs[k] = args[i]
            # call_kwargs.update(kwargs)
            return self.ll(**call_kwargs) * sign

        return cost

    @_needs_data
    def fit(self, verbose=False, **kwargs) -> Tuple[dict, float]:
        """
        Fit the model to the data by maximizing the likelihood.
        Return a dict containing best-fit values of each parameter,
        and the value of the likelihood evaluated there.
        While the optimization is a minimization,
        the likelihood returned is the __maximum__ of the likelihood.

        :param verbose: if True, print the Minuit object
        :type verbose: bool
        :return:
            best-fit values of each parameter,
            and the value of the likelihood evaluated there
        """
        fixed_parameters = list(kwargs.keys())
        guesses = self.parameters.fit_guesses
        guesses.update(kwargs)
        if not self.parameters.values_in_fit_limits(**guesses):
            raise ValueError("Initial guesses are not within fit limits")
        defaults = self.parameters(**guesses)

        cost = self.make_objective(minus=True, **kwargs)

        # Make the Minuit object
        m = Minuit(MinuitWrap(cost, parameters=self.parameters),
                   **defaults)
        m.errordef = Minuit.LIKELIHOOD
        fixed_params = [] if fixed_parameters is None else fixed_parameters
        fixed_params += self.parameters.not_fittable
        for par in fixed_params:
            m.fixed[par] = True

        # Call migrad to do the actual minimization
        m.migrad()
        self.minuit_object = m
        if verbose:
            print(m)
        # alert! This gives the _maximum_ likelihood
        return m.values.to_dict(), -1 * m.fval

    def _confidence_interval_checks(
            self, poi_name: str,
            parameter_interval_bounds: Tuple[float, float],
            confidence_level: float,
            confidence_interval_kind: str,
            **kwargs):
        """
        Helper function for confidence_interval that does the input checks and returns bounds

        :param poi_name: name of the parameter of interest
        :type poi_name: str
        :param parameter_interval_bounds:
            range in which to search for the confidence interval edges
        :type parameter_interval_bounds: Tuple[float, float]
        :param confidence_level: confidence level for confidence intervals
        :type confidence_level: float
        :param confidence_interval_kind: kind of confidence interval to compute
        :type confidence_interval_kind: str
        :return: confidence interval kind, confidence interval threshold, parameter interval bounds
        """
        if confidence_level is None:
            confidence_level = self._confidence_level
        if confidence_interval_kind is None:
            confidence_interval_kind = self._confidence_interval_kind

        if (confidence_level < 0) or (confidence_level > 1):
            raise ValueError("confidence_level must be between 0 and 1")

        parameter_of_interest = self.parameters[poi_name]
        if not parameter_of_interest.fittable:
            raise ValueError("The parameter of interest must be fittable")
        if poi_name in kwargs:
            raise ValueError("You cannot set the parameter you're constraining")

        if parameter_interval_bounds is None:
            parameter_interval_bounds = parameter_of_interest.parameter_interval_bounds
            if parameter_interval_bounds is None:
                raise ValueError(
                    "You must set parameter_interval_bounds in the parameter config"
                    " or when calling confidence_interval")

        if parameter_of_interest.ptype == "rate":
            try:
                if parameter_of_interest.ptype == "rate" and poi_name.endswith("_rate_multiplier"):
                    source_name = poi_name.replace("_rate_multiplier", "")
                else:
                    source_name = poi_name
                mu_parameter = self.get_expectation_values(**kwargs)[source_name]
                parameter_interval_bounds = (
                    parameter_interval_bounds[0] / mu_parameter,
                    parameter_interval_bounds[1] / mu_parameter)
            except NotImplementedError:
                warnings.warn(
                    "The statistical model does not have a get_expectation_values model implemented,"
                    " confidence interval bounds will be set directly.")
                pass  # no problem, continuing with bounds as set

        # define threshold if none is defined:
        if self.confidence_interval_threshold is not None:
            confidence_interval_threshold = self.confidence_interval_threshold
        else:
            # use asymptotic thresholds assuming the test statistic is Chi2 distributed
            if confidence_interval_kind in {"lower", "upper"}:
                critical_value = chi2(1).isf(2 * (1. - confidence_level))
            elif confidence_interval_kind == "central":
                critical_value = chi2(1).isf(1. - confidence_level)

            confidence_interval_threshold = lambda _: critical_value

        return confidence_interval_kind, confidence_interval_threshold, parameter_interval_bounds

    def confidence_interval(
            self, poi_name: str,
            parameter_interval_bounds: Tuple[float, float] = None,
            confidence_level: float = None,
            confidence_interval_kind: str = None,
            **kwargs) -> Tuple[float, float]:
        """
        Uses self.fit to compute confidence intervals for a certain named parameter.
        If the parameter is a rate parameter, and the model has expectation values implemented,
        the bounds will be interpreted as bounds on the expectation value,
        so that the range in the fit is parameter_interval_bounds/mus.
        Otherwise the bound is taken as-is.

        :param poi_name: name of the parameter of interest
        :type poi_name: str
        :param parameter_interval_bounds:
            range in which to search for the confidence interval edges
            May be specified as:
                - setting the property "parameter_interval_bounds" for the parameter
                - passing a list here
        :type parameter_interval_bounds: Tuple[float, float]
        :param confidence_level: confidence level for confidence intervals
        :type confidence_level: float
        :param confidence_interval_kind: kind of confidence interval to compute
        :type confidence_interval_kind: str
        """

        ci_objects = self._confidence_interval_checks(
            poi_name,
            parameter_interval_bounds,
            confidence_level,
            confidence_interval_kind,
            **kwargs)
        confidence_interval_kind, confidence_interval_threshold, parameter_interval_bounds = ci_objects

        # find best-fit:
        best_result, best_ll = self.fit(**kwargs)
        best_parameter = best_result[poi_name]
        mask = (parameter_interval_bounds[0] < best_parameter)
        mask &= (best_parameter < parameter_interval_bounds[1])
        assert mask, ("the best-fit is outside your confidence interval"
            " search limits in parameter_interval_bounds")
        # log-likelihood - critical value:

        # define intersection between likelihood ratio curve and the critical curve:
        def t(hypothesis):
            # define the intersection
            # between the profile-log-likelihood curve and the rejection threshold
            kwargs[poi_name] = hypothesis
            _, ll = self.fit(**kwargs)  # ll is + log-likelihood here
            ret = 2. * (best_ll - ll)  # likelihood curve "right way up" (smiling)
            # if positive, hypothesis is excluded
            return ret - confidence_interval_threshold(hypothesis)

        if confidence_interval_kind in {"upper", "central"}:
            if t(parameter_interval_bounds[1]) > 0:
                ul = brentq(t, best_parameter, parameter_interval_bounds[1])
            else:
                ul = np.inf
        else:
            ul = np.nan

        if confidence_interval_kind in {"lower", "central"}:
            if t(parameter_interval_bounds[0]) > 0:
                dl = brentq(t, parameter_interval_bounds[0], best_parameter)
            else:
                dl = -1 * np.inf
        else:
            dl = np.nan

        return dl, ul


class MinuitWrap:
    """
    Wrapper for functions to be called by Minuit.
    Initialized with a function f and a Parameters instance.

    :param f: function to be wrapped
    :type f: Callable
    :param parameters: parameters of the model
    :type parameters: Parameters
    """

    def __init__(self, f: Callable, parameters: Parameters):
        """Initialize the wrapper"""
        self.func = f
        self.s_args = parameters.names
        self._parameters = {p.name: p.fit_limits for p in parameters}

    def __call__(self, *args):
        return self.func(args)
