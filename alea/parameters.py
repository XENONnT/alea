import numpy as np  # noqa: F401
import scipy.stats as stats  # noqa: F401


class Parameter:
    def __init__(self, name, nominal_value=None, fittable=True,
                 ptype=None, uncertainty=None, relative_uncertainty=None,
                 blueice_anchors=None, fit_limits=None, description=None):
        self.name = name
        self.nominal_value = nominal_value
        self.fittable = fittable
        self.type = ptype
        self._uncertainty = uncertainty
        self.relative_uncertainty = relative_uncertainty
        self.blueice_anchors = blueice_anchors
        self.fit_limits = fit_limits
        self.description = description

    def __repr__(self) -> str:
        parameter_str = [f"{k}={v}" for k, v in self.__dict__.items() if v is not None]
        parameter_str = ", ".join(parameter_str)
        return f'{self.__class__.__module__}.{self.__class__.__qualname__}'\
            f'({parameter_str})'

    @property
    def uncertainty(self):
        if isinstance(self._uncertainty, str):
            # Evaluate the uncertainty if it's a string
            if self._uncertainty.startswith("stats.") or self._uncertainty.startswith("np."):
                return eval(self._uncertainty)
            else:
                raise ValueError(f"Uncertainty string '{self._uncertainty}' must start with 'stats.' or 'np.'")
        else:
            return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value):
        self._uncertainty = value


class Parameters:
    def __init__(self):
        self.parameters = {}

    @classmethod
    def from_config(cls, config):
        parameters = cls()
        for name, param_config in config.items():
            nominal_value = param_config.get("nominal_value", None)
            fittable = param_config.get("fittable", True)
            ptype = param_config.get("ptype", None)
            uncertainty = param_config.get("uncertainty", None)
            relative_uncertainty = param_config.get("relative_uncertainty", None)
            blueice_anchors = param_config.get("blueice_anchors", None)
            fit_limits = param_config.get("fit_limits", None)
            description = param_config.get("description", None)
            parameter = Parameter(name, nominal_value, fittable,
                                  ptype, uncertainty, relative_uncertainty,
                                  blueice_anchors, fit_limits, description)
            parameters.add_parameter(parameter)
        return parameters

    def add_parameter(self, parameter):
        self.parameters[parameter.name] = parameter

    def get_parameter(self, name):
        return self.parameters.get(name)

    @property
    def names(self):
        return list(self.parameters.keys())

    # def fittable_names(self):
    #     return [name for name, param in self.parameters.items() if param.fittable]

    def get_values(self, **kwargs):
        values = {}
        for name, param in self.parameters.items():
            new_val = kwargs.get(name, None)
            values[name] = new_val if new_val is not None else param.nominal_value
        return values

    def __getattr__(self, name):
        if name in self.parameters:
            return self.parameters[name]
        else:
            raise AttributeError(f"Attribute '{name}' not found.")
