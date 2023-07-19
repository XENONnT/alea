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
    def __init__(
            self,
            statistical_model: str,
            statistical_model_args: dict = None,
            parameter_definition: Optional[dict or list] = None,
            confidence_level: float = 0.9,
            confidence_interval_kind: str = 'central',
            confidence_interval_threshold: Callable[[float], float] = None,
            poi: str = None,
            hypotheses: list = None,
            common_generate_values: dict = None,
            true_generate_values: dict = None,
            n_mc: int = 1,
            toydata_file: str = None,
            toydata_mode: str = None,
            metadata: dict = None,
            output_file: str = 'test_toymc.hdf5',
        ):
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
        self.common_generate_values = common_generate_values
        self.true_generate_values = true_generate_values
        self.n_mc = n_mc
        self.toydata_file = toydata_file
        self.toydata_mode = toydata_mode
        self.metadata = metadata if metadata else {}
        self.output_file = output_file

        self.parameter_list, self.result_list, self.result_dtype = self._get_parameter_list()

        self.generate_values = self._get_generate_values()

    def _get_parameter_list(self):
        # parameter_list and result_dtype
        parameter_list = sorted(self.statistical_model.get_parameter_list())
        result_list = parameter_list + ['ll', 'dl', 'ul']
        result_dtype = [(n, float) for n in parameter_list]
        result_dtype += [(n, float) for n in ['ll', 'dl', 'ul']]
        # try:
        #     parameter_list += self.statistical_model.additional_parameters
        #     result_dtype += [(n, float) for n in self.statistical_model.additional_parameters]
        # except:
        #     pass
        return parameter_list, result_list, result_dtype

    def _get_generate_values(self):
        generate_values = []
        hypotheses = deepcopy(self.hypotheses)

        for hypothesis in hypotheses:
            if hypothesis == 'null':
                # there is no signal component
                hypothesis = {self.poi: 0.}
            elif hypothesis == 'true':
                # the true signal component is used
                if self.poi not in self.true_generate_values:
                    raise ValueError(
                        f'{self.poi} should be provided in true_generate_values',
                    )
                hypothesis = {
                    self.poi: self.true_generate_values.get(self.poi),
                }
            elif hypothesis == 'free':
                hypothesis = {}

            array = deepcopy(self.common_generate_values)
            array.update(hypothesis)
            generate_values.append(array)
        return generate_values

    def write_output(self, results):
        metadata = deepcopy(self.metadata)

        result_names = [f'{i:d}' for i in range(len(self.generate_values))]
        for i, ea in enumerate(self.hypotheses):
            if ea in ['null', 'free', 'true']:
                result_names[i] = ea

        metadata['date'] = datetime.now().strftime('%Y%m%d_%H:%M:%S')
        metadata['poi'] = self.poi
        metadata['common_generate_values'] = self.common_generate_values
        metadata['true_generate_values'] = self.true_generate_values

        array_metadatas = [{'generate_values': ea} for ea in self.generate_values]

        numpy_arrays_and_names = [(r, rn) for r, rn in zip(results, result_names)]

        print(f'Saving {self.output_file}')
        numpy_to_toyfile(
            self.output_file,
            numpy_arrays_and_names=numpy_arrays_and_names,
            metadata=metadata,
            array_metadatas=array_metadatas)

    def read_toydata(self):
        toydata, toydata_names = toydata_from_file(self.toydata_file)
        return toydata, toydata_names

    def toy_simulation(self):
        flag_read_toydata = False
        flag_generate_toydata = False
        flag_write_toydata = False

        if self.toydata_mode == 'read':
            flag_read_toydata = True
        elif self.toydata_mode == 'generate':
            flag_generate_toydata = True
        elif self.toydata_mode == 'generate_and_write':
            flag_generate_toydata = True
            flag_write_toydata = True
        elif self.toydata_mode == 'no_toydata':
            pass

        if flag_read_toydata and flag_generate_toydata:
            raise ValueError('Cannot both read and generate toydata')

        toydata = []
        toydata_names = None
        if flag_read_toydata:
            toydata, toydata_names = self.read_toydata()

        results = [np.zeros(self.n_mc, dtype=self.result_dtype) for _ in self.generate_values]
        for i_mc in tqdm(range(self.n_mc)):
            fit_results = []
            for generate_values in self.generate_values:
                if flag_read_toydata:
                    self.statistical_model.data = toydata[i_mc]
                if flag_generate_toydata:
                    self.statistical_model.data = self.statistical_model.generate_data(
                        **generate_values)

                fit_result, max_llh = self.statistical_model.fit(**generate_values)
                fit_result['ll'] = max_llh
                fit_result['dl'] = -1.
                fit_result['ul'] = -1.

                fit_results.append(fit_result)
                toydata.append(self.statistical_model.data)
            for fit_result, result_array in zip(fit_results, results):
                result_array[i_mc] = tuple(fit_result[pn] for pn in self.result_list)

        if flag_write_toydata:
            self.statistical_model.store_data(self.toydata_file, toydata, toydata_names)

        return results

    def run(self):
        results = self.toy_simulation()

        self.write_output(results)
