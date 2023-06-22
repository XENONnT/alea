from alea.statistical_model import StatisticalModel
from alea.simulators import simulate_interpolated
import yaml


class BlueiceExtendedModel(StatisticalModel):
    def __init__(self, parameter_definition: dict, ll_config: dict):
        """
        # TODO write docstring
        """
        super().__init__(parameter_definition=parameter_definition)
        self._ll_blueice = self._build_ll_from_config(ll_config)
        # TODO instead of sefineing self.lls use self._ll.likelihood_list (blueice method)
        # TODO find better name for rgs
        # self.rgs = [simulate_interpolated(ll, binned=b) for ll, b in zip(self.lls, self.binned)]

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def _ll(self, **kwargs):
        # TODO
        # IDEA Set data to blueice ll (or maybe better to set it before explicitly
        # since in the fit this will be called frequently but the data won't change.)
        # IDEA Maybe one could then define self._ll directly in the init instead of _ll_blueice?

        pass

    def _generate_data(self, **kwargs):
        # TODO essentially contains generate_toydata() from ll_nt_from_config
        # IDEA split between science_data and ancillary_measurements?
        pass

    def _generate_science_data(self, **kwargs):
        pass

    def _generate_ancillary_measurements(self, **kwargs):
        pass

    @property
    def nominal_mus(self):
        # TODO
        pass

    def get_mus(self, **kwargs):
        # TODO
        pass

    def _build_ll_from_config(self, ll_config):
        # TODO iterate through ll_config and build blueice ll
        # IDEA Maybe simply return ll in the end and spare the def of _ll?
        # IDEA Or better define a _ll_blueice and call this in _ll to make it more readable?
        ll = None
        return ll
