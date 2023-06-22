from alea.statistical_model import StatisticalModel, data
# from alea.simulators import simulate_interpolated
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
        # TODO analysis_space should be inferred from the data (assert that all sources have the same analysis space)

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

    @data.setter
    def data(self, data):
        """
        Overrides default setter. Will also set the data of the blueice ll.
        Data-sets are expected to be in the form of a list of one or more structured arrays-- representing the data-sets of one or more likelihood terms.
        """
        # TODO set blueice ll data
        self._data = data

    @property
    def nominal_mus(self):
        # TODO
        # IDEA also enable a setter that changes the rate parameters?
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

    def _add_rate_parameters(self):
        # TODO
        # TODO: Check if already set
        pass

    def _add_shape_parameters(self):
        # TODO
        # TODO: Check if already set
        pass
