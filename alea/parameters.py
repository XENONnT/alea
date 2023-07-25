from typing import Any, Dict, List, Optional, Tuple


class Parameter:
    """
    Represents a single parameter with its properties.

    Attributes:
        name (str): The name of the parameter.
        nominal_value (float, optional): The nominal value of the parameter.
        fittable (bool, optional): Indicates if the parameter is fittable or always fixed.
        ptype (str, optional): The ptype of the parameter.
        uncertainty (float or str, optional):
            The uncertainty of the parameter. If a string,
            it can be evaluated as a numpy or scipy function to define non-gaussian constraints.
        relative_uncertainty (bool, optional):
        Indicates if the uncertainty is relative to the nominal_value.
        blueice_anchors (list, optional): Anchors for blueice template morphing.
        fit_limits (tuple, optional): The limits for fitting the parameter.
        parameter_interval_bounds (tupe, optional): limits for computing confidence intervals
        fit_guess (float, optional): The initial guess for fitting the parameter.
        description (str, optional): A description of the parameter.
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
        self.name = name
        self.nominal_value = nominal_value
        self.fittable = fittable
        self.ptype = ptype
        self._uncertainty = uncertainty
        self.relative_uncertainty = relative_uncertainty
        self.blueice_anchors = blueice_anchors
        self.fit_limits = fit_limits
        self.parameter_interval_bounds = parameter_interval_bounds
        self._fit_guess = fit_guess
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
        Returns the uncertainty of the parameter.
        If the uncertainty is a string, it can be evaluated as a numpy or scipy function.
        """
        if isinstance(self._uncertainty, str):
            # Evaluate the uncertainty if it's a string
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
        # make sure to only return fit_guess if fittable
        if self._fit_guess is not None and not self.fittable:
            raise ValueError(
                f"Parameter {self.name} is not fittable, but has a fit_guess.")
        else:
            return self._fit_guess

    @fit_guess.setter
    def fit_guess(self, value: float) -> None:
        self._fit_guess = value

    def __eq__(self, other: object) -> bool:
        """Returns True if all attributes are equal"""
        if isinstance(other, Parameter):
            return all(getattr(self, k) == getattr(other, k) for k in self.__dict__)
        else:
            return False

    def value_in_fit_limits(self, value: float) -> bool:
        """Returns True if value is within fit_limits"""
        if self.fit_limits is None:
            return True
        elif self.fit_limits[0] is None:
            return value <= self.fit_limits[1]
        elif self.fit_limits[1] is None:
            return value >= self.fit_limits[0]
        else:
            return self.fit_limits[0] <= value <= self.fit_limits[1]


class Parameters:
    """
    Represents a collection of parameters.

    Attributes:
        parameters (dict): A dictionary to store the parameters, with parameter name as key.
    """

    def __init__(self):
        self.parameters: Dict[str, Parameter] = {}

    def __iter__(self) -> iter:
        """Returns an iterator over the parameters. Each iteration returns a Parameter object."""
        return iter(self.parameters.values())

    @classmethod
    def from_config(cls, config: Dict[str, dict]):
        """
        Creates a Parameters object from a configuration dictionary.

        Args:
            config (dict): The configuration dictionary.

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
            names (list): List of parameter names.

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
        """
        if parameter.name in self.names:
            raise ValueError(f"Parameter {parameter.name} already exists.")
        self.parameters[parameter.name] = parameter

    @property
    def names(self) -> List[str]:
        """
        Returns a list of parameter names.
        """
        return list(self.parameters.keys())

    @property
    def fit_guesses(self) -> Dict[str, float]:
        """
        Returns a dictionary of fit guesses.
        """
        return {
            name: param.fit_guess
            for name, param in self.parameters.items()
            if param.fit_guess is not None}

    @property
    def fit_limits(self) -> Dict[str, float]:
        """
        Returns a dictionary of fit limits.
        """
        return {
            name: param.fit_limits
            for name, param in self.parameters.items()
            if param.fit_limits is not None}

    @property
    def fittable(self) -> List[str]:
        """
        Returns a list of parameter names which are fittable.
        """
        return [name for name, param in self.parameters.items() if param.fittable]

    @property
    def not_fittable(self) -> List[str]:
        """
        Returns a list of parameter names which are not fittable.
        """
        return [name for name, param in self.parameters.items() if not param.fittable]

    @property
    def uncertainties(self) -> dict:
        """
        return a dict of name:uncertainty for all parameters with a not-NaN uncertainty.
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
        """
        return a dict of name:nominal value for all applicable parameters
        """
        return {
            k: i.nominal_value
            for k, i in self.parameters.items()
            if i.nominal_value is not None}

    def __call__(
            self, return_fittable: bool = False,
            **kwargs: Any) -> Dict[str, float]:
        """
        Returns a dictionary of parameter values, optionally filtered
        to return only fittable parameters.

        Args:
            return_fittable (bool, optional):
                Indicates if only fittable parameters should be returned.
            **kwargs: Additional keyword arguments to override parameter values.

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
                "All parameters must be set explicitly, or have a nominal value,"
                " encountered for: " + emptypars)
        return values

    def __getattr__(self, name: str) -> Parameter:
        """
        Retrieves a Parameter object by attribute access.

        Args:
            name (str): The name of the parameter.

        Returns:
            Parameter: The retrieved Parameter object.

        Raises:
            AttributeError: If the attribute is not found.
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

        Returns:
            Parameter: The retrieved Parameter object.

        Raises:
            KeyError: If the key is not found.
        """
        if name in self.parameters:
            return self.parameters[name]
        else:
            raise KeyError(f"Key '{name}' not found.")

    def __eq__(self, other: object) -> bool:
        """Returns True if all parameters are equal"""
        if isinstance(other, Parameters):
            names = set(self.names + other.names)
            return all(getattr(self, n) == getattr(other, n) for n in names)
        else:
            return False

    def values_in_fit_limits(self, **kwargs: Any) -> bool:
        """
        Returns True if all values are within the fit limits.
        """
        return all(
            self.parameters[name].value_in_fit_limits(value)
            for name, value in kwargs.items())
