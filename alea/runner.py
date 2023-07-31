from copy import deepcopy
from typing import Callable, Optional
from datetime import datetime
from pydoc import locate
import inspect

from tqdm import tqdm
import numpy as np

from inference_interface import toydata_from_file, numpy_to_toyfile
from alea.statistical_model import StatisticalModel


class Runner:
    """
    Runner manipulates statistical model and toydata.
        - initialize statistical model
        - generate or read toy data
        - save toy data if needed
        - fit parameters
        - write output file
    
    One toyfile can contain multiple toydata, but all of them are from the same generate_values

    Attributes:
        statistical_model: statistical model
        poi: parameter of interest
        hypotheses (list): list of hypotheses
        common_hypothesis (dict): common hypothesis, the values are copied to each hypothesis
        generate_values (dict): generate values for toydata
        n_mc (int): number of Monte Carlo
        toydata_file (str): toydata filename
        toydata_mode (str): toydata mode, 'read', 'generate', 'generate_and_write', 'no_toydata'
        metadata (dict): metadata, if None, it is set to {}
        output_file (str): output filename
        result_names (list): list of result names
        result_dtype (list): list of result dtypes
        hypotheses_values (list): list of values for hypotheses

    Args:
        statistical_model (str): statistical model class name
        poi (str): parameter of interest
        hypotheses (list): list of hypotheses
        n_mc (int): number of Monte Carlo
        statistical_model_args (dict, optional (default=None)): arguments for statistical model
        parameter_definition (dict or list, optional (default=None)): parameter definition
        confidence_level (float, optional (default=0.9)): confidence level
        confidence_interval_kind (str, optional (default='central')):
            kind of confidence interval, choice from 'central', 'upper' or 'lower'
        confidence_interval_threshold (Callable[[float], float], optional (default=None)):
            confidence interval threshold of likelihood ratio
        common_hypothesis (dict, optional (default=None)): common hypothesis, the values are copied to each hypothesis
        generate_values (dict, optional (default=None)):
            generate values of toydata. If None, toydata depend on statistical model.
        toydata_mode (str, optional (default='generate_and_write')):
            toydata mode, choice from 'read', 'generate', 'generate_and_write', 'no_toydata'
        toydata_file (str, optional (default=None)): toydata filename
        metadata (dict, optional (default=None)): metadata
        output_file (str, optional (default='test_toymc.hdf5')): output filename

    Todo:
        Implement confidence interval calculation
    """

    def __init__(
            self,
            statistical_model: str,
            poi: str,
            hypotheses: list,
            n_mc: int,
            generate_values: dict = None,
            statistical_model_args: dict = None,
            parameter_definition: Optional[dict or list] = None,
            confidence_level: float = 0.9,
            confidence_interval_kind: str = 'central',
            confidence_interval_threshold: Callable[[float], float] = None,
            common_hypothesis: dict = None,
            toydata_mode: str = 'generate_and_write',
            toydata_file: str = None,
            metadata: dict = None,
            output_file: str = 'test_toymc.hdf5',
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
        self.statistical_model = statistical_model_class(
            parameter_definition=parameter_definition,
            confidence_level=confidence_level,
            confidence_interval_kind=confidence_interval_kind,
            confidence_interval_threshold=confidence_interval_threshold,
            **(statistical_model_args if statistical_model_args else {}),
        )

        self.poi = poi
        self.hypotheses = hypotheses if hypotheses else []
        self.common_hypothesis = common_hypothesis
        self.generate_values = generate_values
        self.n_mc = n_mc
        self.toydata_file = toydata_file
        self.toydata_mode = toydata_mode
        self.metadata = metadata if metadata else {}
        self.output_file = output_file

        self.result_names, self.result_dtype = self._get_parameter_list()

        self.hypotheses_values = self._get_hypotheses()

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
        metadata = deepcopy(self.metadata)

        result_names = [f'{i:d}' for i in range(len(self.hypotheses_values))]
        for i, ea in enumerate(self.hypotheses):
            if ea in ['null', 'free', 'true']:
                result_names[i] = ea

        metadata['date'] = datetime.now().strftime('%Y%m%d_%H:%M:%S')
        metadata['poi'] = self.poi
        metadata['common_hypothesis'] = self.common_hypothesis
        metadata['generate_values'] = self.generate_values

        array_metadatas = [{'hypotheses_values': ea} for ea in self.hypotheses_values]

        numpy_arrays_and_names = [(r, rn) for r, rn in zip(results, result_names)]

        print(f'Saving {self.output_file}')
        numpy_to_toyfile(
            self.output_file,
            numpy_arrays_and_names=numpy_arrays_and_names,
            metadata=metadata,
            array_metadatas=array_metadatas)

    def read_toydata(self):
        """Read toydata from file"""
        toydata, toydata_names = toydata_from_file(self.toydata_file)
        return toydata, toydata_names

    def toy_simulation(self):
        """
        Run toy simulation a specified different toydata mode
        and loop over generate values
        """
        if self.toydata_mode not in {
            'read', 'generate', 'generate_and_write', 'no_toydata',
        }:
            raise ValueError(f'Unknown toydata mode: {self.toydata_mode}')

        toydata = []
        toydata_names = None
        if self.toydata_mode == 'read':
            toydata, toydata_names = self.read_toydata()

        results = [np.zeros(self.n_mc, dtype=self.result_dtype) for _ in self.hypotheses_values]
        for i_mc in tqdm(range(self.n_mc)):
            if self.toydata_mode == 'generate' or self.toydata_mode == 'generate_and_write':
                self.statistical_model.data = self.statistical_model.generate_data(
                    **self.generate_values)
            fit_results = []
            for hypothesis_values in self.hypotheses_values:
                if self.toydata_mode == 'read':
                    self.statistical_model.data = toydata[i_mc]

                fit_result, max_llh = self.statistical_model.fit(**hypothesis_values)
                fit_result['ll'] = max_llh
                fit_result['dl'] = -1.
                fit_result['ul'] = -1.

                fit_results.append(fit_result)
            # appemd toydata
            toydata.append(self.statistical_model.data)
            # assign fitting results
            for fit_result, result_array in zip(fit_results, results):
                result_array[i_mc] = tuple(fit_result[pn] for pn in self.result_names)

        if self.toydata_mode == 'generate_and_write':
            self.statistical_model.store_data(self.toydata_file, toydata, toydata_names)

        return results

    def run(self):
        """Run toy simulation"""
        results = self.toy_simulation()

        self.write_output(results)
