import warnings
from typing import Any, Dict, List, Optional, Tuple

# These imports are needed to evaluate the uncertainty string
import numpy  # noqa: F401
import scipy  # noqa: F401

from alea.utils import within_limits, clip_limits


class Parameter:
    """
    Represents a single parameter with its properties.

    Attributes:
        name (str): The name of the parameter.
        nominal_value (float, optional (default=None)): The nominal value of the parameter.
        fittable (bool, optional (default=None)):
            Indicates if the parameter is fittable or always fixed.
        ptype (str, optional (default=None)): The ptype of the parameter.
        uncertainty (float or str, optional (default=None)): The uncertainty of the parameter.
            If a string, it can be evaluated as a numpy or
            scipy function to define non-gaussian constraints.
        relative_uncertainty (bool, optional (default=None)):
            Indicates if the uncertainty is relative to the nominal_value.
        blueice_anchors (list, optional (default=None)): Anchors for blueice template morphing.
        fit_limits (tuple, optional (default=None)): The limits for fitting the parameter.
        parameter_interval_bounds (tuple, optional (default=None)):
            Limits for computing confidence intervals
        fit_guess (float, optional (default=None)): The initial guess for fitting the parameter.
        description (str, optional (default=None)): A description of the parameter.
    """

    def __init__(
        self,
        name: str,
        nominal_value: Optional[float] = None,
        fittable: bool = True,
        ptype: Optional[str] = None,
        uncertainty: Optional[float or str] = None,
        relative_uncertainty: Optional[bool] = None,
        blueice_anchors: Optional[List] = None,
        fit_limits: Optional[Tuple] = None,
        parameter_interval_bounds: Optional[Tuple] = None,
        fit_guess: Optional[float] = None,
        description: Optional[str] = None,
    ):
        """Initialise a parameter."""
        self.name = name
        self.nominal_value = nominal_value
        self.fittable = fittable
        self.ptype = ptype
        self.uncertainty = uncertainty
        self.relative_uncertainty = relative_uncertainty
        self.blueice_anchors = blueice_anchors
        self.fit_limits = fit_limits
        self.parameter_interval_bounds = parameter_interval_bounds
        self.fit_guess = fit_guess
        self.description = description

    def __repr__(self) -> str:
        parameter_str = [
            f"{k}={v}" for k, v in self.__dict__.items() if v is not None]
        parameter_str = ", ".join(parameter_str)
        _repr = f'{self.__class__.__module__}.{self.__class__.__qualname__}'
        _repr += f'({parameter_str})'
        return _repr

    @property
    def uncertainty(self) -> float or Any:
        """
        Return the uncertainty of the parameter.
        If the uncertainty is a string, it can be evaluated as a numpy or scipy function.
        """
        if isinstance(self._uncertainty, str):
            # Evaluate the uncertainty if it's a string starting with "scipy." or "numpy."
            if self._uncertainty.startswith("scipy.") or self._uncertainty.startswith("numpy."):
                return eval(self._uncertainty)
            else:
                raise ValueError(
                    f"Uncertainty string '{self._uncertainty}'"
                    " must start with 'scipy.' or 'numpy.'")
        else:
            return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value: float or str) -> None:
        self._uncertainty = value

    @property
    def fit_guess(self) -> float:
        """Return the initial guess for fitting the parameter."""
        # make sure to only return fit_guess if fittable
        if self._fit_guess is not None and not self.fittable:
            raise ValueError(
                f"Parameter {self.name} is not fittable, but has a fit_guess.")
        else:
            return self._fit_guess

    @fit_guess.setter
    def fit_guess(self, value: float) -> None:
        self._fit_guess = value

    @property
    def parameter_interval_bounds(self) -> float:
        # make sure to only return parameter_interval_bounds if fittable
        if self._parameter_interval_bounds is not None and not self.fittable:
            raise ValueError(
                f"Parameter {self.name} is not fittable, but has a parameter_interval_bounds.")
        else:
            # print warning when value contains None
            value = self._parameter_interval_bounds
            self._check_parameter_interval_bounds(value)
            return clip_limits(value)

    @parameter_interval_bounds.setter
    def parameter_interval_bounds(self, value: Optional[List]) -> None:
        self._parameter_interval_bounds = value

    def __eq__(self, other: object) -> bool:
        """Return True if all attributes are equal"""
        if isinstance(other, Parameter):
            return all(getattr(self, k) == getattr(other, k) for k in self.__dict__)
        else:
            return False

    def value_in_fit_limits(self, value: float) -> bool:
        """Returns True if value is within fit_limits"""
        return within_limits(value, self.fit_limits)

    def _check_parameter_interval_bounds(self, value):
        """Check if parameter_interval_bounds is within fit_limits and is not None."""
        if (value is None) or (value[0] is None) or (value[1] is None):
            warnings.warn(
                f"parameter_interval_bounds not defined for parameter {self.name}. "
                "This may cause numerical overflow when calculating confidential interval.")
        value = clip_limits(value)
        if not (self.value_in_fit_limits(value[0]) and self.value_in_fit_limits(value[1])):
            raise ValueError(
                f"parameter_interval_bounds {value} not within "
                f"fit_limits {self.fit_limits} for parameter {self.name}.")


class Parameters:
    """
    Represents a collection of parameters.

    Attributes:
        names (List[str]): A list of parameter names.
        fit_guesses (Dict[str, float]): A dictionary of fit guesses.
        fit_limits (Dict[str, float]): A dictionary of fit limits.
        fittable (List[str]): A list of parameter names which are fittable.
        not_fittable (List[str]): A list of parameter names which are not fittable.
        uncertainties (Dict[str, float or Any]): A dictionary of parameter uncertainties.
        with_uncertainty (Parameters): A Parameters object with parameters with
            a not-NaN uncertainty.
        nominal_values (Dict[str, float]): A dictionary of parameter nominal values.
        parameters (Dict[str, Parameter]): A dictionary to store the parameters,
            with parameter name as key.
    """

    def __init__(self):
        """Initialise a collection of parameters."""
        self.parameters: Dict[str, Parameter] = {}

    def __iter__(self) -> iter:
        """Return an iterator over the parameters. Each iteration return a Parameter object."""
        return iter(self.parameters.values())

    @classmethod
    def from_config(cls, config: Dict[str, dict]):
        """
        Creates a Parameters object from a configuration dictionary.

        Args:
            config (dict): A dictionary of parameter configurations.

        Returns:
            Parameters: The created Parameters object.
        """
        parameters = cls()
        for name, param_config in config.items():
            parameter = Parameter(name=name, **param_config)
            parameters.add_parameter(parameter)
        return parameters

    @classmethod
    def from_list(cls, names: List[str]):
        """
        Creates a Parameters object from a list of parameter names.
        Everything else is set to default values.

        Args:
            names (List[str]): List of parameter names.

        Returns:
            Parameters: The created Parameters object.
        """
        parameters = cls()
        for name in names:
            parameter = Parameter(name)
            parameters.add_parameter(parameter)
        return parameters

    def __repr__(self) -> str:
        parameter_str = ", ".join(self.names)
        _repr = f'{self.__class__.__module__}.{self.__class__.__qualname__}'
        _repr += f'({parameter_str})'
        return _repr

    def add_parameter(self, parameter: Parameter) -> None:
        """
        Adds a Parameter object to the Parameters collection.

        Args:
            parameter (Parameter): The Parameter object to add.

        Raises:
            ValueError: If the parameter name already exists.
        """
        if parameter.name in self.names:
            raise ValueError(f"Parameter {parameter.name} already exists.")
        self.parameters[parameter.name] = parameter

    @property
    def names(self) -> List[str]:
        """A list of parameter names."""
        return list(self.parameters.keys())

    @property
    def fit_guesses(self) -> Dict[str, float]:
        """A dictionary of fit guesses."""
        return {
            name: param.fit_guess
            for name, param in self.parameters.items()
            if param.fit_guess is not None}

    @property
    def fit_limits(self) -> Dict[str, float]:
        """A dictionary of fit limits."""
        return {
            name: param.fit_limits
            for name, param in self.parameters.items()
            if param.fit_limits is not None}

    @property
    def fittable(self) -> List[str]:
        """A list of parameter names which are fittable."""
        return [name for name, param in self.parameters.items() if param.fittable]

    @property
    def not_fittable(self) -> List[str]:
        """A list of parameter names which are not fittable."""
        return [name for name, param in self.parameters.items() if not param.fittable]

    @property
    def uncertainties(self) -> dict:
        """
        A dict of uncertainties for all parameters with a not-NaN uncertainty.

        Caution: this is not the same as the parameter.uncertainty property.
        """
        return {k: i.uncertainty for k, i in self.parameters.items() if i.uncertainty is not None}

    @property
    def with_uncertainty(self) -> "Parameters":
        """
        Return parameters with a not-NaN uncertainty.
        The parameters are the same objects as in the original Parameters object, not a copy.
        """
        param_dict = {k: i for k, i in self.parameters.items() if i.uncertainty is not None}
        params = Parameters()
        for param in param_dict.values():
            params.add_parameter(param)
        return params

    @property
    def nominal_values(self) -> dict:
        """A dict of nominal values for all parameters with a nominal value."""
        return {
            k: i.nominal_value
            for k, i in self.parameters.items()
            if i.nominal_value is not None}

    def __call__(
            self, return_fittable: Optional[bool] = False,
            **kwargs: Optional[Dict]) -> Dict[str, float]:
        """
        Return a dictionary of parameter values, optionally filtered
        to return only fittable parameters.

        Args:
            return_fittable (bool, optional (default=False)):
                Indicates if only fittable parameters should be returned.

        Keyword Args:
            kwargs (dict): Additional keyword arguments to override parameter values.

        Raises:
            ValueError: If a parameter name is not found.

        Returns:
            dict: A dictionary of parameter values.
        """
        values = {}

        # check that all kwargs are valid parameter names
        for name in kwargs:
            if name not in self.parameters:
                raise ValueError(f"Parameter '{name}' not found.")

        for name, param in self.parameters.items():
            new_val = kwargs.get(name, None)
            if (return_fittable and param.fittable) or (not return_fittable):
                values[name] = new_val if new_val is not None else param.nominal_value
        if any(i is None for k, i in values.items()):
            emptypars = ", ".join([k for k, i in values.items() if i is None])
            raise AssertionError(
                "All parameters must be set explicitly, or have a nominal value, "
                "not satisfied for: " + emptypars)
        return values

    def __getattr__(self, name: str) -> Parameter:
        """
        Retrieves a Parameter object by attribute access.

        Args:
            name (str): The name of the parameter.

        Raises:
            AttributeError: If the attribute is not found.

        Returns:
            Parameter: The retrieved Parameter object.
        """
        try:
            return super().__getattribute__('parameters')[name]
        except KeyError:
            raise AttributeError(f"Attribute '{name}' not found.")

    def __getitem__(self, name: str) -> Parameter:
        """
        Retrieves a Parameter object by dictionary access.

        Args:
            name (str): The name of the parameter.

        Raises:
            KeyError: If the key is not found.

        Returns:
            Parameter: The retrieved Parameter object.
        """
        if name in self.parameters:
            return self.parameters[name]
        else:
            raise KeyError(f"Key '{name}' not found.")

    def __eq__(self, other: object) -> bool:
        """Return True if all parameters are equal"""
        if isinstance(other, Parameters):
            names = set(self.names + other.names)
            return all(getattr(self, n) == getattr(other, n) for n in names)
        else:
            return False

    def values_in_fit_limits(self, **kwargs: Dict) -> bool:
        """
        Return True if all values are within the fit limits.

        Keyword Args:
            kwargs (dict): The parameter values to check.

        Returns:
            bool: True if all values are within the fit limits.
        """
        return all(
            self.parameters[name].value_in_fit_limits(value)
            for name, value in kwargs.items())
