from copy import deepcopy
from typing import Callable, Optional
from datetime import datetime
from pydoc import locate
import inspect
import warnings

from tqdm import tqdm
import numpy as np

from inference_interface import toydata_from_file, numpy_to_toyfile
from alea.model import StatisticalModel


class Runner:
    """
    Runner manipulates statistical model and toydata.
        - initialize the statistical model
        - generate or reads toy data
        - save toy data if needed
        - fit fittable parameters
        - write the output file
    One toyfile can contain multiple toydata, but all of them are from the same generate_values.

    Attributes:
        statistical_model: statistical model
        poi: parameter of interest
        hypotheses (list): list of hypotheses
        common_hypothesis (dict): common hypothesis, the values are copied to each hypothesis
        generate_values (dict): generate values for toydata
        _compute_confidence_interval (bool): whether compute confidence interval
        _n_mc (int): number of Monte Carlo
        _toydata_file (str): toydata filename
        _toydata_mode (str): toydata mode, 'read', 'generate', 'generate_and_write', 'no_toydata'
        _metadata (dict): metadata, if None, it is set to {}
        _output_file (str): output filename
        _result_names (list): list of result names
        _result_dtype (list): list of result dtypes
        _hypotheses_values (list): list of values for hypotheses

    Args:
        statistical_model (str): statistical model class name
        poi (str): parameter of interest
        hypotheses (list): list of hypotheses
        _n_mc (int): number of Monte Carlo
        statistical_model_args (dict, optional (default={})): arguments for statistical model
        parameter_definition (dict or list, optional (default=None)): parameter definition
        likelihood_config (dict, optional (default=None)): likelihood configuration
        compute_confidence_interval (bool, optional (default=False)):
            whether compute confidence interval
        confidence_level (float, optional (default=0.9)): confidence level
        confidence_interval_kind (str, optional (default='central')):
            kind of confidence interval, choice from 'central', 'upper' or 'lower'
        confidence_interval_threshold (Callable[[float], float], optional (default=None)):
            confidence interval threshold of likelihood ratio
        common_hypothesis (dict, optional (default=None)):
            common hypothesis, the values are copied to each hypothesis
        generate_values (dict, optional (default=None)):
            generate values of toydata. If None, toydata depend on statistical model.
        toydata_mode (str, optional (default='generate_and_write')):
            toydata mode, choice from 'read', 'generate', 'generate_and_write', 'no_toydata'
        toydata_file (str, optional (default=None)): toydata filename
        metadata (dict, optional (default=None)): metadata
        output_file (str, optional (default='test_toymc.h5')): output filename
    """

    def __init__(
            self,
            statistical_model: str,
            poi: str,
            hypotheses: list,
            n_mc: int,
            common_hypothesis: dict = None,
            generate_values: dict = None,
            statistical_model_args: dict = None,
            parameter_definition: Optional[dict or list] = None,
            likelihood_config: dict = None,
            compute_confidence_interval: bool = False,
            confidence_level: float = 0.9,
            confidence_interval_kind: str = 'central',
            confidence_interval_threshold: Callable[[float], float] = None,
            toydata_mode: str = 'generate_and_write',
            toydata_file: str = None,
            metadata: dict = None,
            output_file: str = 'test_toymc.h5',
        ):
        """
        Initialize statistical model,
        parameters list, and generate values list
        """
        statistical_model_class = locate(statistical_model)
        if statistical_model_class is None:
            raise ValueError(f'Could not find {statistical_model}!')
        if not inspect.isclass(statistical_model_class):
            raise ValueError(f'{statistical_model_class} is not a class!')
        if not issubclass(statistical_model_class, StatisticalModel):
            raise ValueError(f'{statistical_model_class} is not a subclass of StatisticalModel!')

        # likelihood_config is keyword argument, because not all statistical model needs it
        if statistical_model_args is None:
            statistical_model_args = {}
        statistical_model_args['likelihood_config'] = likelihood_config
        self.statistical_model = statistical_model_class(
            parameter_definition=parameter_definition,
            confidence_level=confidence_level,
            confidence_interval_kind=confidence_interval_kind,
            confidence_interval_threshold=confidence_interval_threshold,
            **(statistical_model_args if statistical_model_args else {}),
        )

        self.poi = poi
        self.hypotheses = hypotheses if hypotheses else []
        self.common_hypothesis = common_hypothesis if common_hypothesis else {}
        self.generate_values = generate_values if generate_values else {}
        self._compute_confidence_interval = compute_confidence_interval
        self._n_mc = n_mc
        self._toydata_file = toydata_file
        self._toydata_mode = toydata_mode
        self._output_file = output_file
        self._metadata = metadata if metadata else {}

        self._result_names, self._result_dtype = self._get_parameter_list()

        self._hypotheses_values = self._get_hypotheses()

    def _get_parameter_list(self):
        """Get parameter list and result list from statistical model"""
        parameter_list = sorted(self.statistical_model.get_parameter_list())
        # add likelihood, lower limit, and upper limit
        result_names = parameter_list + ['ll', 'dl', 'ul']
        result_dtype = [(n, float) for n in parameter_list]
        result_dtype += [(n, float) for n in ['ll', 'dl', 'ul']]
        return result_names, result_dtype

    def _get_hypotheses(self):
        """Get generate values list from hypotheses"""
        hypotheses_values = []
        hypotheses = deepcopy(self.hypotheses)
        if 'free' not in hypotheses and self._compute_confidence_interval:
            raise ValueError('free hypothesis is needed for confidence interval calculation!')
        if 'free' in hypotheses and hypotheses.index('free') != 0:
            raise ValueError('free hypothesis should be the first hypothesis!')

        for hypothesis in hypotheses:
            if hypothesis == 'null':
                # there is no signal component
                hypothesis = {self.poi: 0.}
            elif hypothesis == 'true':
                # the true signal component is used
                if self.poi not in self.generate_values:
                    raise ValueError(
                        f'{self.poi} should be provided in generate_values',
                    )
                hypothesis = {
                    self.poi: self.generate_values.get(self.poi),
                }
            elif hypothesis == 'free':
                hypothesis = {}

            array = deepcopy(self.common_hypothesis)
            array.update(hypothesis)
            hypotheses_values.append(array)
        return hypotheses_values

    def write_output(self, results):
        """Write output file with metadata"""
        metadata = deepcopy(self._metadata)

        result_names = [f'{i:d}' for i in range(len(self._hypotheses_values))]
        for i, ea in enumerate(self.hypotheses):
            if ea in {'free', 'null', 'true'}:
                result_names[i] = ea

        metadata['date'] = datetime.now().strftime('%Y%m%d_%H:%M:%S')
        metadata['poi'] = self.poi
        metadata['common_hypothesis'] = self.common_hypothesis
        metadata['generate_values'] = self.generate_values

        array_metadatas = [{'hypotheses_values': ea} for ea in self._hypotheses_values]
        numpy_arrays_and_names = [(r, rn) for r, rn in zip(results, result_names)]

        print(f'Saving {self._output_file}')
        numpy_to_toyfile(
            self._output_file,
            numpy_arrays_and_names=numpy_arrays_and_names,
            metadata=metadata,
            array_metadatas=array_metadatas)

    def read_toydata(self):
        """Read toydata from file"""
        toydata, toydata_names = toydata_from_file(self._toydata_file)
        return toydata, toydata_names

    def write_toydata(self, toydata, toydata_names):
        """
        Write toydata to file.
        If toydata is a list of dict, convert it to a list of list.
        """
        self.statistical_model.store_data(self._toydata_file, toydata, toydata_names)

    def data_generator(self):
        """Generate, save or read toydata"""
        # check toydata mode
        if self._toydata_mode not in {
            'read', 'generate', 'generate_and_write', 'no_toydata',
        }:
            raise ValueError(f'Unknown toydata mode: {self._toydata_mode}')
        # check toydata file size
        if self._toydata_mode == 'read':
            toydata, toydata_names = self.read_toydata()
            if len(toydata) < self._n_mc:
                raise ValueError(
                    f'Number of stored toydata {len(toydata)} is '
                    f'less than number of Monte Carlo {self._n_mc}!')
            elif len(toydata) > self._n_mc:
                warnings.warn(
                    f'Number of stored toydata {len(toydata)} is '
                    f'larger than number of Monte Carlo {self._n_mc}.')
        else:
            toydata = []
            toydata_names = None
        # generate toydata
        for i_mc in range(self._n_mc):
            if self._toydata_mode == 'generate' or self._toydata_mode == 'generate_and_write':
                data = self.statistical_model.generate_data(
                    **self.generate_values)
                if self._toydata_mode == 'generate_and_write':
                    # append toydata
                    toydata.append(data)
            elif self._toydata_mode == 'read':
                data = toydata[i_mc]
            elif self._toydata_mode == 'no_toydata':
                data = None
            yield data
        # save toydata
        if self._toydata_mode == 'generate_and_write':
            self.write_toydata(toydata, toydata_names)

    def toy_simulation(self):
        """
        For each Monte Carlo:
            - run toy simulation a specified toydata mode and generate values.
            - loop over hypotheses.

        Todo:
            Implement per-hypothesis switching on whether to compute confidence intervals
        """
        results = [np.zeros(self._n_mc, dtype=self._result_dtype) for _ in self._hypotheses_values]
        for i_mc, data in tqdm(enumerate(self.data_generator()), total=self._n_mc):
            self.statistical_model.data = data
            fit_results = []
            for hypothesis_values in self._hypotheses_values:
                fit_result, max_llh = self.statistical_model.fit(**hypothesis_values)
                fit_result['ll'] = max_llh
                if self._compute_confidence_interval and (self.poi not in hypothesis_values):
                    dl, ul = self.statistical_model.confidence_interval(
                        poi_name=self.poi,
                        best_fit_args=self._hypotheses_values[0],
                        confidence_interval_args=hypothesis_values)
                else:
                    dl, ul = np.nan, np.nan
                fit_result['dl'] = dl
                fit_result['ul'] = ul

                fit_results.append(fit_result)
            # assign fitting results
            for fit_result, result_array in zip(fit_results, results):
                result_array[i_mc] = tuple(fit_result[pn] for pn in self._result_names)
        return results

    def run(self):
        """Run toy simulation"""
        results = self.toy_simulation()

        self.write_output(results)
