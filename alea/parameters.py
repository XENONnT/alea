import warnings
from typing import Any, Dict, List, Tuple, Iterator, Optional, Union, cast
import pandas as pd

from alea.utils import within_limits, clip_limits, evaluate_numpy_scipy_expression


class Parameter:
    """Represents a single parameter with its properties.

    Attributes:
        name (str): The name of the parameter.
        nominal_value (float, optional (default=None)): The nominal value of the parameter.
        fittable (bool, optional (default=True)):
            Indicates if the parameter is fittable or always fixed.
        ptype (str, optional (default=None)): The ptype of the parameter.
        uncertainty (float or str, optional (default=None)): The uncertainty of the parameter.
            If a string, it can be evaluated as a numpy or
            scipy function to define non-gaussian constraints.
        relative_uncertainty (bool, optional (default=None)):
            Indicates if the uncertainty is relative to the nominal_value.
        blueice_anchors (list, optional (default=None)): Anchors for blueice template morphing.
            Blueice will load the template for the provided values and then interpolate
            for any value in between.
        fit_limits (Tuple[float, float], optional (default=None)):
            The limits for fitting the parameter.
        parameter_interval_bounds (Tuple[float, float], optional (default=None)):
            Limits for computing confidence intervals.
        fit_guess (float, optional (default=None)): The initial guess for fitting the parameter.
        description (str, optional (default=None)): A description of the parameter.

    """

    _uncertainty: Optional[Union[float, str]]

    def __init__(
        self,
        name: str,
        nominal_value: Optional[float] = None,
        fittable: bool = True,
        ptype: Optional[str] = None,
        uncertainty: Optional[Union[float, str]] = None,
        relative_uncertainty: Optional[bool] = None,
        blueice_anchors: Optional[Union[list, str]] = None,
        fit_limits: Optional[Tuple] = None,
        parameter_interval_bounds: Optional[Tuple[float, float]] = None,
        fit_guess: Optional[float] = None,
        description: Optional[str] = None,
    ):
        """Initialise a parameter."""
        self.name = name
        self._nominal_value = nominal_value
        self.fittable = fittable
        self.ptype = ptype
        self.relative_uncertainty = relative_uncertainty
        self.uncertainty = uncertainty
        self.blueice_anchors = blueice_anchors
        self.fit_limits = fit_limits
        self.parameter_interval_bounds = parameter_interval_bounds
        self.fit_guess = fit_guess
        self.description = description

        self._check_parameter_consistency()

    def __repr__(self) -> str:
        parameter_str = ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        _repr = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        _repr += f"({parameter_str})"
        return _repr

    @property
    def uncertainty(self) -> Any:
        """Return the uncertainty of the parameter.

        If the uncertainty is a string, it will be evaluated as a numpy or scipy function.

        """
        if isinstance(self._uncertainty, str):
            return evaluate_numpy_scipy_expression(self._uncertainty)
        else:
            return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value: Optional[Union[float, str]]) -> None:
        if self.relative_uncertainty and (value is not None):
            if value and (not isinstance(value, (float, int))):
                raise ValueError(
                    f"When relative_uncertainty of {self.name} is True, "
                    f"uncertainty should be float, not {value}."
                )
            if self.nominal_value is None:
                raise ValueError(
                    f"When relative_uncertainty of {self.name} is True, "
                    "nominal_value should be set."
                )
        self._uncertainty = value

    @property
    def blueice_anchors(self) -> Any:
        """Return the blueice_anchors of the parameter.

        If the blueice_anchors is a string, it will be evaluated as a numpy or scipy function.

        """
        if isinstance(self._blueice_anchors, str):
            return evaluate_numpy_scipy_expression(self._blueice_anchors).tolist()
        else:
            return self._blueice_anchors

    @blueice_anchors.setter
    def blueice_anchors(self, value: Optional[Union[list, str]]) -> None:
        self._blueice_anchors = value

    @property
    def fit_guess(self) -> Optional[float]:
        """Return the initial guess for fitting the parameter."""
        # make sure to only return fit_guess if fittable
        if self._fit_guess is not None and not self.fittable:
            raise ValueError(f"Parameter {self.name} is not fittable, but has a fit_guess.")
        else:
            return self._fit_guess

    @fit_guess.setter
    def fit_guess(self, value: Optional[float]) -> None:
        self._fit_guess = value

    @property
    def parameter_interval_bounds(self) -> Optional[Tuple[float, float]]:
        # make sure to only return parameter_interval_bounds if fittable
        if self._parameter_interval_bounds is not None and not self.fittable:
            raise ValueError(
                f"Parameter {self.name} is not fittable, but has a parameter_interval_bounds."
            )
        else:
            # print warning when value contains None
            value = self._parameter_interval_bounds
            self._check_parameter_interval_bounds(value)
            return clip_limits(value)

    @parameter_interval_bounds.setter
    def parameter_interval_bounds(self, value: Optional[Tuple[float, float]]) -> None:
        self._parameter_interval_bounds = value

    @property
    def nominal_value(self) -> Optional[float]:
        """Return the nominal value of the parameter."""
        return self._nominal_value

    @nominal_value.setter
    def nominal_value(self, value: Optional[float]) -> None:
        if self.needs_reinit and (value != self._nominal_value):
            raise ValueError(
                f"{self.name} is a parameter that requires re-initialization "
                "to change its nominal value "
                f"(tried to override nominal value {self._nominal_value} with {value})."
            )
        self._nominal_value = value

    @property
    def needs_reinit(self) -> bool:
        """Return True if the parameter needs re-initialization (for ptype ``needs_reinit``)."""
        needs_reinit = False
        if self.ptype == "needs_reinit":
            needs_reinit = True
        return needs_reinit

    def __eq__(self, other: object) -> bool:
        """Return True if all attributes are equal."""
        if isinstance(other, Parameter):
            return all(getattr(self, k) == getattr(other, k) for k in self.__dict__)
        else:
            return False

    def value_in_fit_limits(self, value: float) -> bool:
        """Returns True if value is within fit_limits."""
        return within_limits(value, self.fit_limits)

    def _check_parameter_interval_bounds(self, value):
        """Check if parameter_interval_bounds is within fit_limits and is not None."""
        if (value is None) or (value[0] is None) or (value[1] is None):
            warnings.warn(
                f"parameter_interval_bounds not completely defined for parameter {self.name}. "
                "This may cause numerical overflow when calculating confidential interval."
            )
        value = clip_limits(value)
        if not (self.value_in_fit_limits(value[0]) and self.value_in_fit_limits(value[1])):
            raise ValueError(
                f"parameter_interval_bounds {value} not within "
                f"fit_limits {self.fit_limits} for parameter {self.name}."
            )

    def _check_parameter_consistency(self):
        """Check if parameter is consistent."""
        if self.fittable and self.needs_reinit:
            warnings.warn(
                f"Parameter {self.name} is fittable and needs re-initialization. "
                "This may cause unexpected behaviour."
            )
        if (self.blueice_anchors is not None) and self.needs_reinit:
            raise ValueError(
                f"Parameter {self.name} needs re-initialization but has "
                "blueice_anchors defined. "
                "This may cause unexpected behaviour."
            )


class ConditionalParameter:
    """This class is used to define a parameter that depends on another parameter. It has the same
    attributes as the Parameter class but each of them can be a dictionary with keys being the
    values of the conditioning parameter and values being the corresponding values of the
    conditional parameter. Calling the object with the conditioning parameter value as an argument
    will return a corresponding Parameter object with the correct values.

    Attributes:
        name (str): The name of the parameter.
        conditioning_parameter_name (str): The name of the conditioning parameter.

    """

    def __init__(self, name: str, conditioning_parameter_name: str, **kwargs):
        self.name = name
        self.conditioning_name = conditioning_parameter_name
        self.conditions_dict = self._unpack_conditions(kwargs)
        self.conditioning_param = None

    def __repr__(self) -> str:
        parameter_str = ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        _repr = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        _repr += f"({parameter_str})"
        return _repr

    @staticmethod
    def _unpack_conditions(kwargs):
        # 1) collect all condition keys and check for consistency
        all_keys = set()
        for value in kwargs.values():
            if isinstance(value, dict):
                if not all_keys:
                    all_keys = set(value.keys())
                elif all_keys != set(value.keys()):
                    raise ValueError("Inconsistent condition keys across dictionaries.")

        # 2) create the conditions dictionary
        conditions_dict = {key: {} for key in all_keys}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                for condition_key, condition_value in value.items():
                    conditions_dict[condition_key][key] = condition_value
            else:
                for condition_key in all_keys:
                    conditions_dict[condition_key][key] = value

        return conditions_dict

    @property
    def uncertainty(self) -> Any:
        """Return the uncertainty of the parameter (cominal condition)"""
        return self().uncertainty

    @property
    def blueice_anchors(self) -> Any:
        """Return the blueice_anchors of the parameter (cominal condition)"""
        return self().blueice_anchors

    @property
    def fit_guess(self) -> Optional[float]:
        """Return the initial guess for fitting the parameter (cominal condition)"""
        return self().fit_guess

    @property
    def parameter_interval_bounds(self) -> Optional[Tuple[float, float]]:
        """Return the parameter_interval_bounds of the parameter (cominal condition)"""
        return self().parameter_interval_bounds

    @property
    def nominal_value(self) -> Optional[float]:
        """Return the nominal value of the parameter (cominal condition)"""
        return self().nominal_value

    @property
    def needs_reinit(self) -> bool:
        """Return True if the parameter needs re-initialization (for ptype ``needs_reinit``)."""
        return self().needs_reinit

    @property
    def fittable(self) -> bool:
        """Return the fittable attribute of the parameter (cominal condition)"""
        return self().fittable

    @property
    def ptype(self) -> Optional[str]:
        """Return the ptype of the parameter (cominal condition)"""
        return self().ptype

    @property
    def relative_uncertainty(self) -> Optional[bool]:
        """Return the relative_uncertainty of the parameter (cominal condition)"""
        return self().relative_uncertainty

    @property
    def fit_limits(self) -> Optional[Tuple[float, float]]:
        """Return the fit_limits of the parameter (cominal condition)"""
        return self().fit_limits

    def __eq__(self, other: object) -> bool:
        """Return True if all attributes are equal."""
        if isinstance(other, ConditionalParameter):
            return all(getattr(self, k) == getattr(other, k) for k in self.__dict__)
        return False

    def value_in_fit_limits(self, value: float) -> bool:
        """Returns True if value under cominal condition is within fit_limits."""
        return self().value_in_fit_limits(value)

    def __call__(self, **kwargs) -> Parameter:
        if self.conditioning_name in kwargs:
            cond_val = kwargs[self.conditioning_name]
        elif self.conditioning_param is not None:
            cond_val = self.conditioning_param.nominal_value
        else:
            err_msg = (
                f"Conditioning parameter '{self.conditioning_name}' is missing. Can't fall back to "
                "nominal value because conditioning parameter it is not set. "
            )
            raise ValueError(err_msg)
        # check if the conditioning value is in the conditions dictionary
        if cond_val not in self.conditions_dict:
            raise ValueError(
                f"Conditioning value '{cond_val}' not found in the conditions dictionary."
                + f"Available values are: {sorted(list(self.conditions_dict.keys()))}"
            )
        return Parameter(name=self.name, **self.conditions_dict[cond_val])


class Parameters:
    """Represents a collection of parameters.

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
        self.parameters = cast(Dict[str, Parameter], {})

    def __iter__(self) -> Iterator[Parameter]:
        """Return an iterator over the parameters.

        Each iteration return a Parameter object.

        """
        return iter(self.parameters.values())

    @classmethod
    def from_config(cls, config: Dict[str, dict]):
        """Creates a Parameters object from a configuration dictionary.

        Args:
            config (dict): A dictionary of parameter configurations.

        Returns:
            Parameters: The created Parameters object.

        """
        parameters = cls()
        parameter: Union[Parameter, ConditionalParameter]
        for name, param_config in config.items():
            if "conditioning_parameter_name" in param_config:
                parameter = ConditionalParameter(name, **param_config)
            else:
                parameter = Parameter(name=name, **param_config)
            parameters.add_parameter(parameter)
        # set conditioning parameters
        for param in parameters:
            if isinstance(param, ConditionalParameter):
                param.conditioning_param = parameters[param.conditioning_name]
        return parameters

    @classmethod
    def from_list(cls, names: List[str]):
        """Creates a Parameters object from a list of parameter names. Everything else is set to
        default values.

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
        _repr = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        _repr += f"({parameter_str})"
        return _repr

    def __str__(self) -> str:
        """Return an overview table of all parameters."""
        par_list = []
        for p in self:
            if isinstance(p, ConditionalParameter):
                par_dict = {
                    "conditioning_name": p.conditioning_name,
                    "conditions": sorted(p.conditions_dict.keys()),
                }
                # get nominal-condition parameter
                p = p()
            else:
                par_dict = {}
            for k, v in p.__dict__.items():
                # replace hidden attributes with non-hidden properties
                if k.startswith("_"):
                    par_dict[k[1:]] = v
                else:
                    par_dict[k] = v
            par_list.append(par_dict)

        df = pd.DataFrame(par_list)
        # make name column the index
        df.set_index("name", inplace=True)
        df.index.name = None

        return df.to_string()

    def add_parameter(self, parameter: Union[Parameter, ConditionalParameter]) -> None:
        """Adds a Parameter object to the Parameters collection.

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
            if param.fit_guess is not None
        }

    @property
    def fit_limits(self) -> Dict[str, float]:
        """A dictionary of fit limits."""
        return {
            name: param.fit_limits
            for name, param in self.parameters.items()
            if param.fit_limits is not None
        }

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
        """A dict of uncertainties for all parameters with a not-NaN uncertainty.

        Caution: this is not the same as the parameter.uncertainty property.

        """
        return {k: i.uncertainty for k, i in self.parameters.items() if i.uncertainty is not None}

    @property
    def with_uncertainty(self) -> "Parameters":
        """Return parameters with a not-NaN uncertainty.

        The parameters are the same objects as in the original Parameters object, not a copy. For
        conditional parameters, the parameters under the nominal condition are returned.

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
            k: i.nominal_value for k, i in self.parameters.items() if i.nominal_value is not None
        }

    def set_nominal_values(self, **nominal_values):
        """Set the nominal values for parameters.

        Keyword Args:
            nominal_values (dict): A dict of parameter names and values.

        """
        for name, value in nominal_values.items():
            self.parameters[name].nominal_value = value

    def set_fit_guesses(self, **fit_guesses):
        """Set the fit guesses for parameters.

        Keyword Args:
            fit_guesses (dict): A dict of parameter names and values.

        """
        for name, value in fit_guesses.items():
            self.parameters[name].fit_guess = value

    def _evaluate_parameter(self, parameter: Parameter, **kwargs):
        if isinstance(parameter, ConditionalParameter):
            return parameter(**kwargs)
        return parameter

    def __call__(
        self, return_fittable: Optional[bool] = False, **kwargs: Optional[Dict]
    ) -> Dict[str, float]:
        """Return a dictionary of parameter values, optionally filtered to return only fittable
        parameters.

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
            param = self._evaluate_parameter(param, **kwargs)
            new_val = kwargs.get(name, None)
            if param.needs_reinit and new_val != param.nominal_value and new_val is not None:
                raise ValueError(
                    f"{name} is a parameter that requires re-initialization "
                    "to override its nominal value "
                    f"(tried to override nominal value {param.nominal_value} "
                    f"with {new_val})."
                )
            if (return_fittable and param.fittable) or (not return_fittable):
                values[name] = new_val if new_val is not None else param.nominal_value
        if any(i is None for k, i in values.items()):
            emptypars = ", ".join([k for k, i in values.items() if i is None])
            raise AssertionError(
                "All parameters must be set explicitly, or have a nominal value, "
                "not satisfied for: " + emptypars
            )
        return values

    def __getattr__(self, name: str) -> Parameter:
        """Retrieves a Parameter object by attribute access.

        Args:
            name (str): The name of the parameter.

        Raises:
            AttributeError: If the attribute is not found.

        Returns:
            Parameter: The retrieved Parameter object.

        """
        try:
            return super().__getattribute__("parameters")[name]
        except KeyError:
            raise AttributeError(f"Attribute '{name}' not found.")

    def __getitem__(self, name: str) -> Parameter:
        """Retrieves a Parameter object by dictionary access.

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
        """Return True if all parameters are equal."""
        if isinstance(other, Parameters):
            names = set(self.names + other.names)
            return all(getattr(self, n) == getattr(other, n) for n in names)
        else:
            return False

    def values_in_fit_limits(self, **kwargs: Dict) -> bool:
        """Return True if all values are within the fit limits.

        Keyword Args:
            kwargs (dict): The parameter values to check.

        Returns:
            bool: True if all values are within the fit limits.

        """
        for name, value in kwargs.items():
            param = self._evaluate_parameter(self.parameters[name], **kwargs)
            if not param.value_in_fit_limits(value):
                return False
        return True
