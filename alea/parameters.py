from typing import Any, Dict, List, Optional, Tuple
import numpy  # noqa: F401
import scipy  # noqa: F401


class Parameter:
    """
    Represents a single parameter with its properties.

    Attributes:
        name (str): The name of the parameter.
        nominal_value (float, optional): The nominal value of the parameter.
        fittable (bool, optional): Indicates if the parameter is fittable or always fixed.
        ptype (str, optional): The type of the parameter.
        uncertainty (float or str, optional): The uncertainty of the parameter. If a string,
            it can be evaluated as a numpy or scipy function to define non-gaussian constraints.
        relative_uncertainty (bool, optional): Indicates if the uncertainty is relative to the nominal_value.
        blueice_anchors (list, optional): Anchors for blueice template morphing.
        fit_limits (tuple, optional): The limits for fitting the parameter.
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
        fit_guess: Optional[float] = None,
        description: Optional[str] = None,
    ):
        self.name = name
        self.nominal_value = nominal_value
        self.fittable = fittable
        self.type = ptype
        self._uncertainty = uncertainty
        self.relative_uncertainty = relative_uncertainty
        self.blueice_anchors = blueice_anchors
        self.fit_limits = fit_limits
        self.fit_guess = fit_guess
        self.description = description

    def __repr__(self) -> str:
        parameter_str = [f"{k}={v}" for k,
                         v in self.__dict__.items() if v is not None]
        parameter_str = ", ".join(parameter_str)
        return f'{self.__class__.__module__}.{self.__class__.__qualname__}'\
            f'({parameter_str})'

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
                    f"Uncertainty string '{self._uncertainty}' must start with 'scipy.' or 'numpy.'")
        else:
            return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value: float or str) -> None:
        self._uncertainty = value

    def __eq__(self, other: object) -> bool:
        """Returns True if all attributes are equal"""
        if isinstance(other, Parameter):
            return all(getattr(self, k) == getattr(other, k) for k in self.__dict__)
        else:
            return False


class Parameters:
    """
    Represents a collection of parameters.

    Attributes:
        parameters (dict): A dictionary to store the parameters, with parameter name as key.
    """

    def __init__(self):
        self.parameters: Dict[str, Parameter] = {}

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
            nominal_value = param_config.get("nominal_value", None)
            fittable = param_config.get("fittable", True)
            ptype = param_config.get("ptype", None)
            uncertainty = param_config.get("uncertainty", None)
            relative_uncertainty = param_config.get(
                "relative_uncertainty", None)
            blueice_anchors = param_config.get("blueice_anchors", None)
            fit_limits = param_config.get("fit_limits", None)
            fit_guess = param_config.get("fit_guess", None)
            description = param_config.get("description", None)
            parameter = Parameter(
                name=name,
                nominal_value=nominal_value,
                fittable=fittable,
                ptype=ptype,
                uncertainty=uncertainty,
                relative_uncertainty=relative_uncertainty,
                blueice_anchors=blueice_anchors,
                fit_limits=fit_limits,
                fit_guess=fit_guess,
                description=description
            )
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

    def add_parameter(self, parameter: Parameter) -> None:
        """
        Adds a Parameter object to the Parameters collection.

        Args:
            parameter (Parameter): The Parameter object to add.
        """
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
        return {name: param.fit_guess for name, param in self.parameters.items() if param.fit_guess is not None}

    @property
    def fit_limits(self) -> Dict[str, float]:
        """
        Returns a dictionary of fit limits.
        """
        return {name: param.fit_limits for name, param in self.parameters.items() if param.fit_limits is not None}

    @property
    def fittable(self) -> List[str]:
        """
        Returns a list of parameter names which are fittable.
        """
        return [name for name, param in self.parameters.items() if param.fittable]

    def __call__(self, return_fittable: bool = False,
                 **kwargs: Any) -> Dict[str, float]:
        """
        Returns a dictionary of parameter values, optionally filtered
        to return only fittable parameters.

        Args:
            return_fittable (bool, optional): Indicates if only fittable parameters should be returned.
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
            if (return_fittable and param.fittable) or not return_fittable:
                values[name] = new_val if new_val is not None else param.nominal_value
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
        if name in self.parameters:
            return self.parameters[name]
        else:
            raise AttributeError(f"Attribute '{name}' not found.")

    def __eq__(self, other: object) -> bool:
        """Returns True if all parameters are equal"""
        if isinstance(other, Parameters):
            names = set(self.names + other.names)
            return all(getattr(self, n) == getattr(other, n) for n in names)
        else:
            return False
