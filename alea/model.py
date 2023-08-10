import inspect
import warnings
from copy import deepcopy
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
from scipy.stats import chi2
from scipy.optimize import brentq
from iminuit import Minuit
from blueice.likelihood import _needs_data
from inference_interface import toydata_to_file

from alea.parameters import Parameters
from alea.utils import within_limits, clip_limits


class StatisticalModel:
    """
    Class that defines a statistical model.

    - The statisical model contains two parts that you must define yourself:
        - a likelihood function
            ll(self, parameter_1, parameter_2... parameter_n):
            A function of a set of named parameters which
            return a float expressing the loglikelihood for observed data given these parameters.
        - a data generation function
            generate_data(self, parameter_1, parameter_2... parameter_n):
            A function of the same set of named parameters return a full data set.
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
        - confidence_interval

    The public methods generate_data and ll, as the names suggested,
    depend on private methods _generate_data, and _ll respectively.

    Attributes:
        data: data of the model
        _data: data of the model
        _confidence_level: confidence level for confidence intervals
        _confidence_interval_kind: kind of confidence interval to compute
        parameters: parameters of the model
        confidence_interval_threshold: threshold for confidence interval
        is_data_set (bool): True if data is set

    Args:
        data: pre-set data of the model
        parameter_definition (dict or list, optional (default=None)):
            definition of the parameters of the model
        confidence_level (float, optional (default=0.9)):
            confidence level for confidence intervals
        confidence_interval_kind (str, optional (default="central")):
            kind of confidence interval to compute
        confidence_interval_threshold (Callable[[float], float], optional (default=None)):
            threshold for confidence interval

    Raise:
        RuntimeError: if you try to instantiate the StatisticalModel class directly
        NotImplementedError: if you do not implement the likelihood function or the data generation
    """

    def __init__(
            self,
            data = None,
            parameter_definition: Optional[dict or list] = None,
            confidence_level: float = 0.9,
            confidence_interval_kind: str = "central",  # one of central, upper, lower
            confidence_interval_threshold: Callable[[float], float] = None,
            **kwargs,
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
        """Likelihood function, return the loglikelihood for the given parameters."""
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

        Keyword Args:
            kwargs: keyword arguments for the parameters

        Returns:
            float: likelihood value
        """
        parameters = self.parameters(**kwargs)
        return self._ll(**parameters)

    def generate_data(self, **kwargs) -> dict or list:
        """
        Generate data for the given parameters.
        The parameters are passed as keyword arguments, positional arguments are not possible.
        If a parameter is not given, the default value is used.

        Raises:
            ValueError: If the parameters are not within the fit limits

        Returns:
            dict or list: generated data

        Caution:
            This implementation won't allow you to call generate_data by positional arguments.
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
            file_name, data_list,
            data_name_list: Optional[List] = None,
            metadata: Optional[Dict] = None):
        """
        Store a list of datasets.
        (each on the form of a list of one or more structured arrays or dicts)
        Using inference_interface, but included here to allow over-writing.
        The structure would be: [[datasets1], [datasets2], ..., [datasetsn]],
        where each of datasets is a list of structured arrays.
        If you specify, it is set, if not it will read from self.get_likelihood_term_names.
        If not defined, it will be ["0", "1", ..., "n-1"]. The metadata is optional.

        Args:
            file_name (str): name of the file to store the data in
            data_list (list): list of datasets
            data_name_list (list, optional (default=None)): list of names of the datasets.
                If None, it will be read from self.get_likelihood_term_names
            metadata (dict, optional (default=None)): metadata to store with the data.
                If None, no metadata is stored.
        """
        if all([isinstance(d, dict) for d in data_list]):
            _data_list = [list(d.values()) for d in data_list]
        elif all([isinstance(d, list) for d in data_list]):
            _data_list = data_list
        else:
            raise ValueError(
                'Unsupported mixed toydata format! '
                'toydata should be a list of dict or a list of list',)

        if data_name_list is None:
            if hasattr(self, "likelihood_names"):
                data_name_list = self.likelihood_names
            else:
                data_name_list = ["{:d}".format(i) for i in range(len(_data_list[0]))]

        kw = {'metadata': metadata} if metadata is not None else dict()
        if len(_data_list[0]) != len(data_name_list):
            raise ValueError(
                "The number of data sets and data names must be the same")
        toydata_to_file(file_name, _data_list, data_name_list, **kw)

    def get_expectation_values(self, **parameter_values):
        """
        Get the expectation values of the measurement.

        Args:
            parameter_values: values of the parameters
        """
        return NotImplementedError("get_expectation_values is optional to implement")

    @property
    def nominal_expectation_values(self):
        """
        Nominal expectation values for the sources of the likelihood.

        For this to work, you must implement `get_expectation_values`.
        """
        return self.get_expectation_values()  # no kwargs for nominal

    def get_likelihood_term_from_name(self, likelihood_name: str) -> int:
        """
        Return the index of a likelihood term if the likelihood has several names

        Args:
            likelihood_name (str): name of the likelihood term

        Returns:
            int: index of the likelihood term
        """
        if hasattr(self, "likelihood_names"):
            likelihood_names = self.likelihood_names
            return {n: i for i, n in enumerate(likelihood_names)}[likelihood_name]
        else:
            raise NotImplementedError("The attribute likelihood_names is not defined.")

    def get_parameter_list(self):
        """Return a set of all parameters that the generate_data and likelihood accepts"""
        return self.parameters.names

    def make_objective(self):
        """
        Make a function that can be passed to Minuit

        Returns:
            Callable: function that can be passed to Minuit
        """
        def cost(args):
            # Get the arguments from args
            call_kwargs = {}
            for i, k in enumerate(self.parameters.names):
                call_kwargs[k] = args[i]
            # for optimization, we want to minimize the negative log-likelihood
            return self.ll(**call_kwargs) * -1

        return cost

    @_needs_data
    def fit(self, verbose=False, **kwargs) -> Tuple[dict, float]:
        """
        Fit the model to the data by maximizing the likelihood.
        Return a dict containing best-fit values of each parameter,
        and the value of the likelihood evaluated there.
        While the optimization is a minimization,
        the likelihood returned is the __maximum__ of the likelihood.

        Args:
            verbose (bool): if True, print the Minuit object

        Returns:
            dict, float: best-fit values of each parameter,
            and the value of the likelihood evaluated there
        """
        fixed_parameters = list(kwargs.keys())
        guesses = self.parameters.fit_guesses
        guesses.update(kwargs)
        if not self.parameters.values_in_fit_limits(**guesses):
            raise ValueError("Initial guesses are not within fit limits")
        defaults = self.parameters(**guesses)

        cost = self.make_objective()

        # Make the Minuit object
        m = Minuit(MinuitWrap(cost, parameters=self.parameters), **defaults)
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
        Helper function for confidence_interval that does the input checks and return bounds

        Args:
            poi_name (str): name of the parameter of interest
            parameter_interval_bounds (Tuple[float, float]): range in which to search for the
                confidence interval edges
            confidence_level (float): confidence level for confidence intervals
            confidence_interval_kind (str): kind of confidence interval to compute

        Returns:
            Tuple[str, Callable[[float], float], Tuple[float, float]]:
                confidence interval kind, confidence interval threshold, parameter interval bounds
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
        else:
            value = parameter_interval_bounds
            parameter_of_interest._check_parameter_interval_bounds(value)
            parameter_interval_bounds = clip_limits(value)

        if parameter_of_interest.ptype == "rate":
            try:
                if parameter_of_interest.ptype == "rate" and poi_name.endswith("_rate_multiplier"):
                    source_name = poi_name.replace("_rate_multiplier", "")
                else:
                    source_name = poi_name
                mu_parameter = self.get_expectation_values(**kwargs)[source_name]
                # update parameter_interval_bounds because poi is rate_multiplier
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
            confidence_interval_args: dict = None,
            best_fit_args: dict = None,
        ) -> Tuple[float, float]:
        """
        Uses self.fit to compute confidence intervals for a certain named parameter.
        If the parameter is a rate parameter, and the model has expectation values implemented,
        the bounds will be interpreted as bounds on the expectation value,
        so that the range in the fit is parameter_interval_bounds/mus.
        Otherwise the bound is taken as-is.

        Args:
            poi_name (str): name of the parameter of interest
            parameter_interval_bounds (Tuple[float, float], optional (default=None)): range
                in which to search for the confidence interval edges. May be specified as:
                    - setting the property "parameter_interval_bounds" for the parameter
                    - passing a list here
                    - passing None here, in which case the parameter_interval_bounds property of the parameter is used
            confidence_level (float, optional (default=None)):
                confidence level for confidence intervals.
                If None, the default confidence level of the model is used.
            confidence_interval_kind (str, optional (default=None)):
                kind of confidence interval to compute.
                If None, the default kind of the model is used.

        Keyword Args:
            confidence_interval_args (dict, optional (default=None)): Parameters that will be fixed
                in the profile likelihood computation. If None, all fittable parameters will be profiled except the poi
            best_fit_args (dict, optional (default=None)): If you require the "global" best-fit used to normalise the
                profile likelihood ratio to fix fewer parameters than the profile likelihood-- mainly used for 1-D slices
                of higher-dimensional confidence volumes, where the global best-fit may not be along the profile.
                if None, will be set to confidence_interval_args
        """
        if confidence_interval_args is None:
            confidence_interval_args = {}
        if best_fit_args is None:
            best_fit_args = confidence_interval_args
        ci_objects = self._confidence_interval_checks(
            poi_name,
            parameter_interval_bounds,
            confidence_level,
            confidence_interval_kind,
            **confidence_interval_args)
        confidence_interval_kind, confidence_interval_threshold, parameter_interval_bounds = ci_objects

        # best_fit_args only provides the best-fit likelihood
        _, best_ll = self.fit(**best_fit_args)
        # the optimization of profile-likelihood under
        # confidence_interval_args provides the best_parameter
        best_result, _ = self.fit(**confidence_interval_args)
        best_parameter = best_result[poi_name]
        mask = within_limits(best_parameter, parameter_interval_bounds)
        if not mask:
            raise ValueError(
                f"The best-fit {best_parameter} is outside your confidence interval "
                f"search limits in parameter_interval_bounds {parameter_interval_bounds}.")

        # define intersection between likelihood ratio curve and the critical curve:
        def t(hypothesis_value):
            # define the intersection
            # between the profile-log-likelihood curve and the rejection threshold
            _confidence_interval_args = deepcopy(confidence_interval_args)
            _confidence_interval_args[poi_name] = hypothesis_value
            _, ll = self.fit(**_confidence_interval_args)  # ll is + log-likelihood here
            ret = 2. * (best_ll - ll)  # likelihood curve "right way up" (smiling)
            # if positive, hypothesis is excluded
            return ret - confidence_interval_threshold(hypothesis_value)

        t_best_parameter = t(best_parameter)

        if t_best_parameter > 0:
            warnings.warn(f"CL calculation failed, given fixed parameters {confidence_interval_args}.")

        if confidence_interval_kind in {"upper", "central"} and t_best_parameter < 0:
            if t(parameter_interval_bounds[1]) > 0:
                ul = brentq(t, best_parameter, parameter_interval_bounds[1])
            else:
                ul = np.inf
        else:
            ul = np.nan

        if confidence_interval_kind in {"lower", "central"} and t_best_parameter < 0:
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

    Attributes:
        func: function wrapped
        s_args (list): parameter names of the model
        _parameters (dict): parameters and limits of the model

    Args:
        f (Callable): function to be wrapped
        parameters (Parameters): parameters of the model
    """

    def __init__(self, f: Callable, parameters: Parameters):
        """Initialize the wrapper"""
        self.func = f
        self.s_args = parameters.names
        self._parameters = {p.name: p.fit_limits for p in parameters}

    def __call__(self, *args):
        return self.func(args)
