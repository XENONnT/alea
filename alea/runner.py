import time
import inspect
from copy import deepcopy
from typing import Optional, Dict, Union
from datetime import datetime
import warnings

from tqdm import tqdm
import numpy as np
from inference_interface import toydata_from_file, numpy_to_toyfile

from alea.model import StatisticalModel
from alea.utils import load_yaml


class Runner:
    """Runner manipulates statistical model and toydata.

        - initialize the statistical model
        - generate or reads toy data
        - save toy data if needed
        - fit fittable parameters
        - write the output file
    One toyfile can contain multiple toydata, but all of them are from the same generate_values.

    Attributes:
        model (StatisticalModel): statistical model instance
        poi (str): parameter of interest
        hypotheses (list): list of hypotheses
        common_hypothesis (dict): common hypothesis, the values are copied to each hypothesis
        generate_values (dict): generate values for toydata
        nominal_values (dict): nominal values of parameters
        _compute_confidence_interval (bool): whether compute confidence interval
        _n_mc (int): number of Monte Carlo
        _toydata_filename (str): toydata filename
        _toydata_mode (str): toydata mode, 'read', 'generate', 'generate_and_store', 'no_toydata'
        _metadata (dict): metadata, if None, it is set to {}
        _output_filename (str): output filename
        _result_names (list): list of result names
        _result_dtype (list): list of result dtypes
        _hypotheses_values (list): list of values for hypotheses

    Args:
        statistical_model (str): statistical model class name
        poi (str): parameter of interest
        hypotheses (list): list of hypotheses
        n_mc (int): number of Monte Carlo
        common_hypothesis (dict, optional (default=None)):
            common hypothesis, the values are copied to each hypothesis
        generate_values (Dict[str, float], optional (default=None)):
            generate values of toydata. If None, toydata depend on statistical model.
        nominal_values (dict, optional (default=None)):
            nominal values of parameters. If None, nothing will be assigned to model.
        statistical_model_config (str, optional (default=None)):
            statistical model configuration filename
        parameter_definition (dict or list, optional (default=None)): parameter definition
        statistical_model_args (dict, optional (default={})): arguments for statistical model
        likelihood_config (dict, optional (default=None)): likelihood configuration
        compute_confidence_interval (bool, optional (default=False)):
            whether compute confidence interval
        confidence_level (float, optional (default=0.9)): confidence level
        confidence_interval_kind (str, optional (default='central')):
            kind of confidence interval, choice from 'central', 'upper' or 'lower'
        fit_strategy (dict, optional (default=None)): fit strategy dictionary.
            If None, the default fit strategy of the model will be used.
        toydata_mode (str, optional (default='generate_and_store')):
            toydata mode, choice from 'read', 'generate', 'generate_and_store', 'no_toydata'
        toydata_filename (str, optional (default=None)): toydata filename
        only_toydata (bool, optional (default=False)): whether only generate toydata
        output_filename (str, optional (default='test_toymc.ii.h5')): output filename
        seed (int, optional (default=None)): random seed for runners before generating toydata
        metadata (dict, optional (default=None)): metadata to be saved in output file

    """

    def __init__(
        self,
        statistical_model: str = "alea.examples.gaussian_model.GaussianModel",
        poi: str = "mu",
        hypotheses: list = ["free"],
        n_mc: int = 1,
        common_hypothesis: Optional[dict] = None,
        generate_values: Optional[Dict[str, float]] = None,
        nominal_values: Optional[dict] = None,
        statistical_model_config: Optional[str] = None,
        parameter_definition: Optional[Union[dict, list]] = None,
        statistical_model_args: Optional[dict] = None,
        likelihood_config: Optional[dict] = None,
        compute_confidence_interval: bool = False,
        confidence_level: float = 0.9,
        confidence_interval_kind: str = "central",
        fit_strategy: Optional[dict] = None,
        toydata_mode: str = "generate_and_store",
        toydata_filename: str = "test_toydata_filename.ii.h5",
        only_toydata: bool = False,
        output_filename: str = "test_output_filename.ii.h5",
        seed: Optional[int] = None,
        metadata: Optional[dict] = None,
    ):
        """Initialize statistical model, parameters list, and generate values list."""
        self.poi = poi

        statistical_model_class = StatisticalModel.get_model_from_name(statistical_model)

        # if statistical_model_config is provided
        # overwrite parameter_definition and likelihood_config
        if statistical_model_config is not None:
            model_config = load_yaml(statistical_model_config)
            if parameter_definition is not None:
                raise ValueError(
                    "parameter_definition is duplicated, "
                    "because statistical_model_config is provided!"
                )
            if likelihood_config is not None:
                raise ValueError(
                    "likelihood_config is duplicated, "
                    "because statistical_model_config is provided!"
                )
            parameter_definition = model_config["parameter_definition"]
            likelihood_config = model_config["likelihood_config"]
            # in case of fit_strategy is provided in both
            # statistical_model_config and arguments fit_strategy in
            # arguments will overwrite fit_strategy in statistical_model_config
            fit_strategy = {**model_config.get("fit_strategy", {}), **(fit_strategy or {})}

        # update nominal_values into statistical_model_args
        if statistical_model_args is None:
            statistical_model_args = {}
        # nominal_values is keyword argument
        self.nominal_values = nominal_values if nominal_values else {}
        # initialize nominal_values only once
        statistical_model_args["nominal_values"] = self.nominal_values
        # likelihood_config is keyword argument, because not all statistical model needs it
        statistical_model_args["likelihood_config"] = likelihood_config

        # initialize statistical model
        self.model = statistical_model_class(
            parameter_definition=parameter_definition,
            confidence_level=confidence_level,
            confidence_interval_kind=confidence_interval_kind,
            fit_strategy=fit_strategy,
            **statistical_model_args,
        )

        self.hypotheses = hypotheses if hypotheses else []
        self.common_hypothesis = common_hypothesis if common_hypothesis else {}
        self.generate_values = generate_values if generate_values else {}
        self._compute_confidence_interval = compute_confidence_interval
        self._n_mc = n_mc
        self._toydata_filename = toydata_filename
        self._toydata_mode = toydata_mode
        self._output_filename = output_filename
        self.only_toydata = only_toydata
        self.seed = seed
        self._metadata = metadata if metadata else {}

        self._result_names, self._result_dtype = self._get_parameter_list()

        self._hypotheses_values = self._get_hypotheses()

        # find confidence_interval_thresholds function for the hypotheses
        from alea.submitters.local import NeymanConstructor

        self.confidence_interval_thresholds = NeymanConstructor.get_confidence_interval_thresholds(
            self.poi,
            self._hypotheses_values,
            statistical_model_args.get("limit_threshold", None),
            nominal_values,
            confidence_interval_kind,
            confidence_level,
            statistical_model_args.get("limit_threshold_interpolation", False),
            statistical_model_args.get("asymptotic_dof", 1),
        )

    def pre_process_poi(self, value, attribute_name):
        """Pre-process of poi_expectation for some attributes of runner."""
        if not all([isinstance(v, (float, int)) for v in value.values()]):
            raise ValueError(
                f"{attribute_name} should be a dict of float! But {value} is provided."
            )
        # update poi according to poi_expectation
        if "poi_expectation" in value:
            value = self.update_poi(self.model, self.poi, value, self.nominal_values)
        return value

    @property
    def generate_values(self) -> Dict[str, float]:
        return self._generate_values

    @generate_values.setter
    def generate_values(self, value: Dict[str, float]) -> None:
        if "poi_expectation" in value:
            self.input_poi_expectation = True
        else:
            self.input_poi_expectation = False
        # update poi according to poi_expectation
        self._generate_values = self.pre_process_poi(value, "generate_values")

    @property
    def common_hypothesis(self) -> Dict[str, float]:
        return self._common_hypothesis

    @common_hypothesis.setter
    def common_hypothesis(self, value: Dict[str, float]) -> None:
        # update poi according to poi_expectation
        self._common_hypothesis = self.pre_process_poi(value, "common_hypothesis")

    @property
    def hypotheses(self) -> list:
        return self._hypotheses

    @hypotheses.setter
    def hypotheses(self, values: list) -> None:
        # update poi according to poi_expectation
        for i in range(len(values)):
            if isinstance(values[i], dict):
                values[i] = self.pre_process_poi(values[i], "hypothesis")
        self._hypotheses = values

    @staticmethod
    def runner_arguments():
        """Get runner arguments and annotations."""
        # find run toyMC default args and annotations:
        # reference: https://docs.python.org/3/library/inspect.html#inspect.getfullargspec
        (
            args,
            varargs,
            varkw,
            defaults,
            kwonlyargs,
            kwonlydefaults,
            annotations,
        ) = inspect.getfullargspec(Runner.__init__)
        # skip the first one because it is self(Runner itself)
        default_args = dict(zip(args[1:], defaults))
        return args, default_args, annotations

    @staticmethod
    def update_poi(
        model, poi: str, generate_values: Dict[str, float], nominal_values: Dict[str, float] = {}
    ):
        """Update the poi according to poi_expectation. First, it will check if poi_expectation is
        provided, if not so, it will do nothing. Second, it will check if poi is provided, if so, it
        will raise error. Third, it will check if poi ends with _rate_multiplier, if not so, it will
        raise error. Finally, it will update poi to the correct value according to poi_expectation
        using the get_expectation_values method of model, under specified nominal_values.

        Args:
            poi (str): parameter of interest
            generate_values (dict): generate values of toydata,
                it can contain "poi_expectation"
            nominal_values (dict): nominal values of parameters

        Caution:
            The expectation is evaluated under nominal_values in each batch.

        """
        if poi in generate_values:
            raise ValueError(
                f"You can not specify both {poi} "
                f"along with poi_expectation, "
                f"because {poi} will be updated according to poi_expectation."
            )
        if not poi.endswith("_rate_multiplier"):
            raise ValueError(
                f"poi {poi} should end with _rate_multiplier, "
                "if poi_expectation is provided, because you want to update "
                "the generate_values according to the expectations."
            )
        generate_values_copy = deepcopy(generate_values)
        generate_values_copy.pop("poi_expectation")
        expectation_values = model.get_expectation_values(
            **{**generate_values_copy, **nominal_values}
        )
        component = poi.replace("_rate_multiplier", "")
        nominal_expectation = expectation_values[component]
        poi_expectation = generate_values["poi_expectation"]
        ratio = poi_expectation / nominal_expectation
        # update poi to the correct value
        generate_values.pop("poi_expectation")
        generate_values[poi] = ratio
        return generate_values

    def _get_parameter_list(self):
        """Get parameter list and result list from statistical model."""
        parameter_list = sorted(self.model.get_parameter_list())
        # add likelihood, lower limit, upper limit, and the migrad valid fit bool
        result_names = parameter_list + ["ll", "dl", "ul"]
        result_dtype = [(n, float) for n in result_names]
        result_names += ["valid_fit"]
        result_dtype += [("valid_fit", bool)]
        return result_names, result_dtype

    def _get_hypotheses(self):
        """Get generate values list from hypotheses.

        Caution:
            When free hypothesis is provided, it should be the first hypothesis.
            Free hypothesis means that all parameters are free to fit, it will
            not use common_hypothesis!

        """
        allowed_hypothesis_strs = ["zero", "true", "free"]
        hypotheses_values = []
        hypotheses = deepcopy(self.hypotheses)
        if len(hypotheses) == 0:
            raise ValueError("hypotheses should not be empty!")
        if "free" not in hypotheses and self._compute_confidence_interval:
            warnings.warn(
                f"If free hypothesis is not provided for confidence interval calculation, "
                f"the first hypothesis {hypotheses[0]} will be used for confidence "
                f"interval calculation."
            )
        if "free" in hypotheses and hypotheses.index("free") != 0:
            raise ValueError("free hypothesis should be the first hypothesis!")

        for hypothesis in hypotheses:
            # translate hypothesis
            if hypothesis == "free":
                # if free hypothesis, will not use common_hypothesis
                h = {}
            elif hypothesis == "zero":
                # there is no poi
                h = deepcopy(self.common_hypothesis)
                h.update({self.poi: 0.0})
            elif hypothesis == "true":
                # the true poi is used
                if self.poi not in self.generate_values:
                    raise ValueError(
                        f"{self.poi} should be provided in generate_values",
                    )
                h = deepcopy(self.common_hypothesis)
                h.update(
                    {
                        self.poi: self.generate_values.get(self.poi),
                    }
                )
            else:
                if not isinstance(hypothesis, dict):
                    raise ValueError(
                        f"If str hypothesis is not {allowed_hypothesis_strs}, it should be a dict!"
                    )
                h = deepcopy(self.common_hypothesis)
                h.update(hypothesis)

            if not all([isinstance(v, (float, int)) for v in h.values()]):
                raise ValueError(f"hypothesis should be a dict of float! But {h} is provided.")
            hypotheses_values.append(h)

        # check if the length of hypotheses and hypotheses_values are the same
        if len(hypotheses) != len(hypotheses_values):
            raise ValueError(
                "Something wrong with the length of hypotheses and hypotheses_values, "
                "please check the code!"
            )
        return hypotheses_values

    def write_output(self, results):
        """Write output file with metadata."""
        metadata = deepcopy(self._metadata)

        n_hypo = len(self._hypotheses_values)
        result_names = [f"{i:0{len(str(n_hypo))}d}" for i in range(n_hypo)]
        for i, ea in enumerate(self.hypotheses):
            if isinstance(ea, str) and (ea in {"free", "zero", "true"}):
                result_names[i] = ea

        metadata["date"] = datetime.now().strftime("%Y%m%d_%H:%M:%S")
        metadata["poi"] = self.poi
        metadata["common_hypothesis"] = self.common_hypothesis
        metadata["generate_values"] = self.generate_values
        metadata["nominal_values"] = self.nominal_values
        metadata["seed"] = self.seed
        try:
            metadata["expectation_values"] = self.model.get_expectation_values(
                **self.generate_values
            )
        except NotImplementedError:
            metadata["expectation_values"] = {}

        array_metadatas = [{"hypotheses_values": ea} for ea in self._hypotheses_values]
        numpy_arrays_and_names = [(r, rn) for r, rn in zip(results, result_names)]

        print(f"Saving {self._output_filename}")
        numpy_to_toyfile(
            self._output_filename,
            numpy_arrays_and_names=numpy_arrays_and_names,
            metadata=metadata,
            array_metadatas=array_metadatas,
        )

    def read_toydata(self):
        """Read toydata from file."""
        toydata, toydata_names = toydata_from_file(self._toydata_filename)
        return toydata, toydata_names

    def store_toydata(self, toydata, toydata_names):
        """Write toydata to file.

        If toydata is a list of dict, convert it to a list of list.

        """
        print(f"Saving {self._toydata_filename}")
        self.model.store_data(self._toydata_filename, toydata, toydata_names)

    def data_generator(self):
        """Generate, save or read toydata."""
        # set seed
        np.random.seed(self.seed)
        # check toydata mode
        if self._toydata_mode not in {
            "read",
            "generate",
            "generate_and_store",
            "no_toydata",
        }:
            raise ValueError(f"Unknown toydata mode: {self._toydata_mode}")
        if self.only_toydata and self._toydata_mode != "generate_and_store":
            raise ValueError(
                f"only_toydata is True, you should only generate_and_store, "
                f"but toydata_mode is {self._toydata_mode}!"
            )
        # check toydata file size
        if self._toydata_mode == "read":
            toydata, toydata_names = self.read_toydata()
            if len(toydata) < self._n_mc:
                raise ValueError(
                    f"Number of stored toydata {len(toydata)} is "
                    f"less than number of Monte Carlo {self._n_mc}!"
                )
            elif len(toydata) > self._n_mc:
                warnings.warn(
                    f"Number of stored toydata {len(toydata)} is "
                    f"larger than number of Monte Carlo {self._n_mc}."
                )
        else:
            toydata = []
            toydata_names = None
        # generate toydata
        for i_mc in range(self._n_mc):
            if self._toydata_mode == "generate" or self._toydata_mode == "generate_and_store":
                # generate toydata
                data = self.model.generate_data(**self.generate_values)
                # set fit guesses as generate values
                self.model.set_fit_guesses(**self.generate_values)
                if self._toydata_mode == "generate_and_store":
                    # append toydata
                    toydata.append(data)
            elif self._toydata_mode == "read":
                data = toydata[i_mc]
            elif self._toydata_mode == "no_toydata":
                data = None
            yield data
        # save toydata
        if self._toydata_mode == "generate_and_store":
            self.store_toydata(toydata, toydata_names)

    def simulate(self):
        """Only generate toydata."""
        all(tqdm(self.data_generator(), total=self._n_mc))

    def simulate_and_fit(self):
        """
        For each Monte Carlo:
            - run toy simulation a specified toydata mode and generate values.
            - loop over hypotheses.

        Todo:
            Implement per-hypothesis switching on whether to compute confidence intervals
        """
        results = [np.zeros(self._n_mc, dtype=self._result_dtype) for _ in self._hypotheses_values]
        for i_mc, data in tqdm(enumerate(self.data_generator()), total=self._n_mc):
            self.model.data = data
            fit_results = []
            for i_hypo, hypothesis_values in enumerate(self._hypotheses_values):
                fit_result, max_llh = self.model.fit(**hypothesis_values)
                fit_result["ll"] = max_llh
                fit_result["valid_fit"] = self.model.minuit_object.valid

                if self._compute_confidence_interval and (self.poi not in hypothesis_values):
                    # hypothesis_values should only be a fittable subset of parameters
                    non_fittable = set(hypothesis_values.keys()) - set(
                        self.model.parameters.fittable
                    )
                    if non_fittable:
                        raise ValueError(
                            f"The hypothesis {hypothesis_values} "
                            f"should only be a subset of the fittable parameters "
                            f"{self.model.parameters.fittable} in the statistical model. "
                            f"Because the non-fittable parameters {non_fittable} are fixed "
                            f"to nominal values in 'free' hypothesis."
                        )
                    dl, ul = self.model.confidence_interval(
                        poi_name=self.poi,
                        best_fit_args=self._hypotheses_values[0],
                        confidence_interval_args=hypothesis_values,
                        confidence_interval_threshold=self.confidence_interval_thresholds[i_hypo],
                    )
                else:
                    dl, ul = np.nan, np.nan
                fit_result["dl"] = dl
                fit_result["ul"] = ul

                fit_results.append(fit_result)
            # assign fitting results
            for fit_result, result_array in zip(fit_results, results):
                result_array[i_mc] = tuple(fit_result[pn] for pn in self._result_names)
        return results

    def run(self):
        """Run toy simulation.

        If only_toydata is True, only generate toydata.

        """
        global_start = time.time()
        cpu_global_start = time.process_time()
        if self.only_toydata:
            self.simulate()
        else:
            results = self.simulate_and_fit()
            self.write_output(results)
        print(
            "Used real time {0:.02f}s, CPU time {1:.02f}s".format(
                time.time() - global_start, time.process_time() - cpu_global_start
            )
        )
        print("Alea iacta est.")
