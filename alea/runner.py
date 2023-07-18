from copy import deepcopy
from pydoc import locate
from typing import Callable, Optional

from tqdm import tqdm
import numpy as np


class Runner:
    def __init__(
            self,
            model_name: str,
            parameter_definition: Optional[dict or list] = None,
            likelihood_config: dict = None,
            confidence_level: float = 0.9,
            confidence_interval_kind: str = "central",  # one of central, upper, lower
            confidence_interval_threshold: Callable[[float], float] = None,
            poi: str = None,
            hypotheses: list = [{}],
            common_generate_values: dict={},
            true_generate_values: dict={},
            guess: dict={},
            n_mc: int = 10,
            toydata_file: str = None,
            toydata_mode: str = None,
            metadata: dict = None,
            output_file: str='test_toymc.hdf5',
            **kwargs):
        model_class = locate(model_name)
        self.model = model_class(
            parameter_definition,
            likelihood_config,
            confidence_level,
            confidence_interval_kind,
            confidence_interval_threshold,
            **kwargs,
        )

        self.poi = poi
        self.hypotheses = hypotheses
        self.common_generate_values = common_generate_values
        self.true_generate_values = true_generate_values
        self.n_mc = n_mc
        self.toydata_file = toydata_file
        self.toydata_mode = toydata_mode
        self.metadata = metadata
        self.output_file = output_file
        self.guess = guess

        self.parameter_list, self.result_dtype = self.get_parameter_list()

        self.generate_values = self.get_generate_values()

    def get_parameter_list(self):
        # parameter_list and result_dtype
        parameter_list = sorted(self.model.get_parameter_list())
        result_dtype = [(n, float) for n in parameter_list]
        try:
            parameter_list += self.model.additional_parameters
            result_dtype += [(n, float) for n in self.model.additional_parameters]
        except:
            pass
        return parameter_list, result_dtype

    def get_generate_values(self):
        arrays = []
        hypotheses = deepcopy(self.hypotheses)

        for hypothesis in hypotheses:
            if hypothesis == 'null':
                # there is no signal component
                hypothesis = {self.poi: 0.}
            elif hypothesis == 'true':
                # the true signal component is used
                if self.poi not in self.true_generate_values:
                    raise ValueError(
                        f"{self.poi} should be provided in true_generate_values",
                    )
                hypothesis = {
                    self.poi: self.true_generate_values.get(self.poi)
                }
            elif hypothesis == 'free':
                hypothesis = {}

            array = deepcopy(self.common_generate_values)
            array.update(hypothesis)
            arrays.append(array)
        return arrays

    def write_toydata(self):
        self.model.write_toydata()
        if self.inference_class_name == 'binference.likelihoods.ll_GOF.InferenceObject':
            self.model.write_reference()

    def write_output(self, results):
        metadata = deepcopy(self.metadata)
        result_names = self.result_names

        # TODO: this might not be needed
        if (self.extra_args[0] == 'iterate') and (result_names is None):
            result_names = ['{:.3f}'.format(float(v)) for v in self.extra_args[2]]

        if result_names is None:
            # TODO: whether just save dict as string?
            result_names = [f'{i:d}' for i in range(len(self.extra_args))]
            for i, ea in enumerate(
                    self.extra_args
                ):  #if using named extra args (free, null, true), use that name
                if ea in ['null', 'free', 'true']:
                    result_names[i] = ea

        metadata['date'] = datetime.now().strftime('%Y%m%d_%H:%M:%S')
        metadata['generate_args'] = self.generate_args
        metadata['signal_expectation'] = self.signal_expectation
        metadata['signal_component_name'] = self.signal_component_name
        metadata['nominal_expectations'] = self.nominal_expectations

        # TODO: this might not be needed
        if self.extra_args[0] == 'iterate':
            eas = [{self.extra_args[1]: v} for v in self.extra_args[2]]
            array_metadatas = [{'extra_args': ea} for ea in eas]
        else:
            array_metadatas = [{'extra_args': ea} for ea in self.extra_args]

        numpy_arrays_and_names = [(r, rn) for r, rn in zip(results, result_names)]

        print(f'Saving {self.output_filename}')
        numpy_to_toyfile(
            self.output_filename,
            numpy_arrays_and_names=numpy_arrays_and_names,
            metadata=metadata,
            array_metadatas=array_metadatas)

    def toy_simulation(self):
        results = [np.zeros(self.n_mc, dtype=self.result_dtype) for _ in self.generate_values]
        for i in tqdm(range(self.n_mc)):
            fit_results = self.model.
            for fit_result, result_array in zip(fit_results, results):
                fit_result_array = tuple(fit_result[pn] for pn in self.parameter_list)
                result_array[i] = fit_result_array
        return results

    def run(self, n_mc):
        simple_data = self.model.generate_data()
        self.model.store_data('simple_data', [simple_data])

    def run_toymcs(self):
        results = self.toy_simulation()

        if self.toydata_mode == 'write':
            self.write_toydata()

        self.write_output(results)
